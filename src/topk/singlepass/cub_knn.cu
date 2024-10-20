#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cub/block/block_radix_sort.cuh>

#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/topk/singlepass/cub_knn.hpp"

#include "bits/topk/singlepass/detail/definitions_common.hpp"

#include "bits/topk/bitonic_sort_static.cuh"

namespace
{

template <std::size_t K, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ float broadcast_radius(float (&reg_dists)[ITEMS_PER_THREAD], float* shm)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto element_idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (element_idx == K - 1)
        {
            *shm = reg_dists[i];
        }
    }

    __syncthreads();

    return *shm;
}

template <std::size_t K, std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD,
          typename SortAlgorithm>
__device__ __forceinline__ float
merge_buffer(SortAlgorithm& sort, float (&reg_dists)[ITEMS_PER_THREAD],
             std::int32_t (&reg_labels)[ITEMS_PER_THREAD], knn::pair_t* buffer,
             std::int32_t* buffer_size, float* current_radius)
{
    const auto local_buffer_size = *buffer_size;

    // initialize register arrays for buffer values
    float buffer_dists[ITEMS_PER_THREAD];
    std::int32_t buffer_labels[ITEMS_PER_THREAD];

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        buffer_dists[i] = std::numeric_limits<float>::infinity();
    }

// load buffer values from shared memory
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = i * BLOCK_SIZE + threadIdx.x;
        if (idx < local_buffer_size && idx < K)
        {
            const auto value = buffer[idx];
            buffer_dists[i] = value.distance;
            buffer_labels[i] = value.index;
        }
    }

    // sort the buffer in descending order
    sort.SortDescending(buffer_dists, buffer_labels);

    __syncthreads();

// bitonic merge the top k list and buffer
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        if (reg_dists[i] > buffer_dists[i])
        {
            reg_dists[i] = buffer_dists[i];
            reg_labels[i] = buffer_labels[i];
        }
    }

    // sort the top k list
    sort.Sort(reg_dists, reg_labels);

    __syncthreads();

    // reset buffer size
    if (threadIdx.x == 0)
    {
        *buffer_size = std::max<std::int32_t>(*buffer_size - K, 0);
    }

    // broadcast radius to all threads
    // broadcast contains __syncthreads() so we don't have to do it here
    return broadcast_radius<K>(reg_dists, current_radius);
}

template <std::size_t K, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
__global__ void cub_kernel(array_view<float, 2> input, array_view<float, 2> out_dist,
                           array_view<std::int32_t, 2> out_label)
{
    // number of items per thread
    constexpr std::size_t ITEMS_PER_THREAD = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    using sort_t = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, std::int32_t>;

    __shared__ float current_radius;
    __shared__ knn::pair_t buffer[K];
    __shared__ std::int32_t buffer_size;
    __shared__ typename sort_t::TempStorage sort_storage;

    sort_t sort_alg{sort_storage};

    // initialize buffer size
    if (threadIdx.x == 0)
    {
        buffer_size = 0;
    }

    // initialize buffer
    for (std::size_t i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        buffer[i].distance = std::numeric_limits<float>::infinity();
    }

    __syncthreads();

    // top k list split among the threads in the block
    // thread i has elements a_{i * ITEMS_PER_THREAD}, a_{i * ITEMS_PER_THREAD + 1}, ...
    float reg_dists[ITEMS_PER_THREAD];
    std::int32_t reg_labels[ITEMS_PER_THREAD];

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        reg_dists[i] = std::numeric_limits<float>::infinity();
    }

    const auto input_count = input.size(1);

// load the first block from global memory
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto point_idx = i * BLOCK_SIZE + threadIdx.x;
        if (point_idx < input_count && point_idx < K)
        {
            reg_dists[i] = input(blockIdx.x, point_idx);
            reg_labels[i] = point_idx;
        }
    }

    // sort the first block
    sort_alg.Sort(reg_dists, reg_labels);

    __syncthreads();

    // initialize the radius to the kth element
    auto radius = broadcast_radius<K>(reg_dists, &current_radius);

    float dist[BATCH_SIZE];
    std::int32_t label[BATCH_SIZE];
    std::int32_t buffer_pos[BATCH_SIZE];

    // while at least one thread in the block will read a value in the next iteration
    for (auto i = K + threadIdx.x; i < input_count + threadIdx.x; i += BLOCK_SIZE * BATCH_SIZE)
    {
// load the next batch of distances
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            const auto point_idx = i + j * BLOCK_SIZE;

            dist[j] = std::numeric_limits<float>::infinity();
            buffer_pos[j] = -1; // none (do not add this item to the buffer)

            // read the next value from input
            if (point_idx < input_count)
            {
                dist[j] = input(blockIdx.x, point_idx);
                label[j] = point_idx;
            }
        }

// allocate buffer positions
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            // if this pair is a candidate for top k, allocate a buffer position for it
            if (dist[j] < radius)
            {
                buffer_pos[j] = atomicAdd(&buffer_size, 1);
            }
        }

        // merge the buffer if necessary
        for (;;)
        {
            bool overflown = false;

#pragma unroll
            for (std::size_t j = 0; j < BATCH_SIZE; ++j)
            {
                // try to add the loaded value to the buffer (negative position indicates no value)
                if (0 <= buffer_pos[j] && buffer_pos[j] < K)
                {
                    buffer[buffer_pos[j]].distance = dist[j];
                    buffer[buffer_pos[j]].index = label[j];
                }

                // check for buffer overflow
                overflown |= buffer_pos[j] >= static_cast<std::int32_t>(K);
                // update buffer position for the next iteration
                buffer_pos[j] -= K;
            }

            // we don't have to merge if the buffer is not full
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            radius = merge_buffer<K, BLOCK_SIZE>(sort_alg, reg_dists, reg_labels, buffer,
                                                 &buffer_size, &current_radius);
        }
    }

    radius = merge_buffer<K, BLOCK_SIZE>(sort_alg, reg_dists, reg_labels, buffer, &buffer_size,
                                         &current_radius);

// copy the values to the output
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto index = threadIdx.x * ITEMS_PER_THREAD + i;
        if (index < K)
        {
            out_dist(blockIdx.x, index) = reg_dists[i];
            out_label(blockIdx.x, index) = reg_labels[i];
        }
    }
}

template <std::size_t K, std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ float broadcast_radius_striped(float (&reg_dists)[ITEMS_PER_THREAD],
                                                          float* shm)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD / 2; ++i)
    {
        const auto element_idx = i * BLOCK_SIZE + threadIdx.x;
        if (element_idx == K - 1)
        {
            *shm = reg_dists[i];
        }
    }

    __syncthreads();

    return *shm;
}

template <std::size_t K, std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD,
          typename SortAlgorithm>
__device__ __forceinline__ float
merge_buffer_direct(SortAlgorithm& sort, float (&reg_dists)[ITEMS_PER_THREAD],
                    std::int32_t (&reg_labels)[ITEMS_PER_THREAD], knn::pair_t* buffer,
                    std::int32_t* buffer_size, float* current_radius)
{
#pragma unroll
    for (std::size_t i = ITEMS_PER_THREAD / 2; i < ITEMS_PER_THREAD; ++i)
    {
        reg_dists[i] = std::numeric_limits<float>::infinity();
    }

// load buffer values from shared memory
#pragma unroll
    for (std::size_t i = ITEMS_PER_THREAD / 2; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = (i - ITEMS_PER_THREAD / 2) * BLOCK_SIZE + threadIdx.x;
        if (idx < *buffer_size && idx < K)
        {
            const auto value = buffer[idx];
            reg_dists[i] = value.distance;
            reg_labels[i] = value.index;
        }
    }

    // sort the buffer in descending order
    sort.SortBlockedToStriped(reg_dists, reg_labels);

    __syncthreads();

    // reset buffer size
    if (threadIdx.x == 0)
    {
        *buffer_size = std::max<std::int32_t>(*buffer_size - K, 0);
    }

    // broadcast radius to all threads
    return broadcast_radius_striped<K, BLOCK_SIZE>(reg_dists, current_radius);
}

/** k-selection kernel using cub radix sort directly (i.e., sort both blocks at once)
 */
template <std::size_t K, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
__global__ void cub_direct_kernel(array_view<float, 2> input, array_view<float, 2> out_dist,
                                  array_view<std::int32_t, 2> out_label)
{
    // number of items per thread (we want to have space for total of 2K items for the top k list
    // and buffer)
    constexpr std::size_t ITEMS_PER_THREAD = 2 * (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    using sort_t = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, std::int32_t>;

    __shared__ knn::pair_t buffer[K];
    __shared__ std::int32_t buffer_size;
    __shared__ typename sort_t::TempStorage sort_storage;
    __shared__ float current_radius;

    sort_t sort_alg{sort_storage};

    // initialize buffer size
    if (threadIdx.x == 0)
    {
        buffer_size = 0;
    }

    // initialize buffer
    for (std::size_t i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        buffer[i].distance = std::numeric_limits<float>::infinity();
    }

    __syncthreads();

    // top k list striped among the threads in the block
    float reg_dists[ITEMS_PER_THREAD];
    std::int32_t reg_labels[ITEMS_PER_THREAD];

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        reg_dists[i] = std::numeric_limits<float>::infinity();
    }

    const auto input_count = input.size(1);

// load the first block from global memory
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD / 2; ++i)
    {
        const auto point_idx = i * BLOCK_SIZE + threadIdx.x;
        if (point_idx < input_count && point_idx < K)
        {
            reg_dists[i] = input(blockIdx.x, point_idx);
            reg_labels[i] = point_idx;
        }
    }

    // sort the first block
    sort_alg.SortBlockedToStriped(reg_dists, reg_labels);

    // initialize the radius to the kth element
    auto radius = broadcast_radius_striped<K, BLOCK_SIZE>(reg_dists, &current_radius);

    float dist[BATCH_SIZE];
    std::int32_t label[BATCH_SIZE];
    std::int32_t buffer_pos[BATCH_SIZE];

    // while at least one thread in the block will read a value in the next iteration
    for (auto i = K + threadIdx.x; i < input_count + threadIdx.x; i += BLOCK_SIZE * BATCH_SIZE)
    {
// load the next batch of distances
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            const auto point_idx = i + j * BLOCK_SIZE;

            dist[j] = std::numeric_limits<float>::infinity();
            buffer_pos[j] = -1; // none (do not add this item to the buffer)

            // read the next value from input
            if (point_idx < input_count)
            {
                dist[j] = input(blockIdx.x, point_idx);
                label[j] = point_idx;
            }
        }

// allocate buffer positions
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            // if this pair is a candidate for top k, allocate a buffer position for it
            if (dist[j] < radius)
            {
                buffer_pos[j] = atomicAdd(&buffer_size, 1);
            }
        }

        // merge the buffer if necessary
        for (;;)
        {
            bool overflown = false;

#pragma unroll
            for (std::size_t j = 0; j < BATCH_SIZE; ++j)
            {
                // try to add the loaded value to the buffer (negative position indicates no value)
                if (0 <= buffer_pos[j] && buffer_pos[j] < K)
                {
                    buffer[buffer_pos[j]].distance = dist[j];
                    buffer[buffer_pos[j]].index = label[j];
                }

                // check for buffer overflow
                overflown |= buffer_pos[j] >= static_cast<std::int32_t>(K);
                // update buffer position for the next iteration
                buffer_pos[j] -= K;
            }

            // we don't have to merge if the buffer is not full
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            radius = merge_buffer_direct<K, BLOCK_SIZE>(sort_alg, reg_dists, reg_labels, buffer,
                                                        &buffer_size, &current_radius);
        }
    }

    radius = merge_buffer_direct<K, BLOCK_SIZE>(sort_alg, reg_dists, reg_labels, buffer,
                                                &buffer_size, &current_radius);

// copy the values to the output
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD / 2; ++i)
    {
        const auto index = i * BLOCK_SIZE + threadIdx.x;
        if (index < K)
        {
            out_dist(blockIdx.x, index) = reg_dists[i];
            out_label(blockIdx.x, index) = reg_labels[i];
        }
    }
}

} // namespace

void cub_knn::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    constexpr std::size_t BATCH_SIZE = 8;

    if (!dynamic_switch<TOPK_SINGLEPASS_K_VALUES>(k(), [&]<std::size_t K>() {
            constexpr std::size_t THREADS_PER_BLOCK = (K <= 256) ? 64 : 128;
            cub_kernel<K, THREADS_PER_BLOCK, BATCH_SIZE>
                <<<block_count, THREADS_PER_BLOCK>>>(dist, out_dist, out_label);
        }))
    {
        throw std::runtime_error("Unsupported k value: " + std::to_string(k()));
    }

    cuda_stream::make_default().sync();
}

void cub_direct::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    constexpr std::size_t BATCH_SIZE = 8;

    if (!dynamic_switch<TOPK_SINGLEPASS_K_VALUES>(k(), [&]<std::size_t K>() {
            constexpr std::size_t THREADS_PER_BLOCK = (K <= 32) ? 32 : 128;
            cub_direct_kernel<K, THREADS_PER_BLOCK, BATCH_SIZE>
                <<<block_count, THREADS_PER_BLOCK>>>(dist, out_dist, out_label);
        }))
    {
        throw std::runtime_error("Unsupported k value: " + std::to_string(k()));
    }

    cuda_stream::make_default().sync();
}
