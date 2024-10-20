#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/topk/singlepass/partial_bitonic_buffered.hpp"

#include "bits/topk/singlepass/detail/definitions_common.hpp"

#include "bits/ptx_utils.cuh"
#include "bits/topk/bitonic_sort.cuh"
#include "bits/topk/bitonic_sort_static.cuh"

namespace
{

template <bool USE_WARP_SORT, typename Layout>
__device__ __forceinline__ void load_block(Layout block, std::int32_t base_point_idx,
                                           array_view<float, 2> input, std::size_t k)
{
    const auto input_count = input.size(1);

    // assumes k is a multiple of 32 (since the whole warp has to participate on warp_sort)
    for (std::int32_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        // load value from global memory
        auto point_idx = base_point_idx + i;
        auto dist = point_idx >= input_count ? std::numeric_limits<float>::infinity()
                                             : input(blockIdx.x, point_idx);

        // sort the loaded values in this warp
        if (USE_WARP_SORT)
        {
            warp_sort(dist, point_idx);
        }

        // store the sorted subsequences to shared memory
        block.dist(i) = dist;
        block.label(i) = point_idx;
    }

    __syncthreads();
}

/** Merge unsorted buffer with a sorted block of top k pairs
 *
 * @param block Block of top k pairs (sorted by distance)
 * @param buffer Buffer of unsorted items
 * @param buffer_size Number of elements in the buffer
 * @param k Maximal number of elements in @p block and @p buffer
 *
 * @returns new radius after the merge operation
 */
template <bool USE_WARP_SORT>
__device__ float merge_buffer(knn::pair_t* __restrict__ block, knn::pair_t* __restrict__ buffer,
                              std::int32_t* buffer_size, std::size_t k)
{
    // sort the buffer
    block_sort<USE_WARP_SORT, 1, order_t::ascending, aos_layout>({buffer}, k);
    // merge it with current top k list
    block_merge<USE_WARP_SORT, order_t::ascending, aos_layout>({block}, k, 2 * k);
    // sort the lower half of the values
    block_sort_bitonic<USE_WARP_SORT, order_t::ascending, aos_layout>({block}, k / 2, k);

    // reset buffer size
    if (threadIdx.x == 0)
    {
        *buffer_size = std::max<std::int32_t>(*buffer_size - k, 0);
    }

    // update current radius
    auto radius = block[k - 1].distance;

    // reset buffer positions for the next call
    for (std::size_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        buffer[i].distance = std::numeric_limits<float>::infinity();
    }

    __syncthreads();

    return radius;
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE>
__device__ float merge_buffer(knn::pair_t* __restrict__ block, knn::pair_t* __restrict__ buffer,
                              std::int32_t* buffer_size)
{
    // sort the buffer
    block_sort<USE_WARP_SORT, 1, K, BLOCK_SIZE, aos_layout>({buffer});
    // merge it with current top k list
    block_merge<USE_WARP_SORT, K, K * 2, BLOCK_SIZE, aos_layout>({block});
    // sort the lower half of the values
    block_sort_bitonic<USE_WARP_SORT, K / 2, K, BLOCK_SIZE, aos_layout>({block});

    // reset buffer size
    if (threadIdx.x == 0)
    {
        *buffer_size = std::max<std::int32_t>(*buffer_size - K, 0);
    }

    // update current radius
    auto radius = block[K - 1].distance;

// reset buffer positions for the next call
#pragma unroll
    for (std::size_t i = 0; i < K; i += BLOCK_SIZE)
    {
        const auto idx = i + threadIdx.x;
        if (idx < K)
        {
            buffer[i].distance = std::numeric_limits<float>::infinity();
        }
    }

    __syncthreads();

    return radius;
}

template <bool USE_WARP_SORT>
__global__ void buffered_partial_bitonic_aos(array_view<float, 2> input,
                                             array_view<float, 2> out_dist,
                                             array_view<std::int32_t, 2> out_label, std::size_t k)
{
    constexpr std::size_t PRESORTED_SIZE = USE_WARP_SORT ? 32 : 1;

    extern __shared__ std::uint8_t shm[];
    __shared__ std::int32_t buffer_size;

    // split shared memory to two blocks
    knn::pair_t* block_a = reinterpret_cast<knn::pair_t*>(shm);
    knn::pair_t* block_b = block_a + k;

    if (threadIdx.x == 0)
    {
        buffer_size = 0;
    }

    const auto input_count = input.size(1);

    // reset buffer
    for (std::size_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        block_b[i].distance = std::numeric_limits<float>::infinity();
    }

    // load and sort the first block
    load_block<USE_WARP_SORT, aos_layout>({block_a}, 0, input, k);
    block_sort<USE_WARP_SORT, PRESORTED_SIZE, order_t::ascending, aos_layout>({block_a}, k);

    // initialize the radius to the kth element
    auto radius = block_a[k - 1].distance;

    // while at least one thread in the block will read a value in the next iteration
    for (auto i = k + threadIdx.x; i < input_count + threadIdx.x; i += blockDim.x)
    {
        auto local_dist = std::numeric_limits<float>::infinity();
        std::int32_t local_label = 0;
        std::int32_t buffer_pos = -1; // none (do not add this item to the buffer)

        // read the next value from input
        if (i < input_count)
        {
            local_dist = input(blockIdx.x, i);
            local_label = i;
        }

        // if this pair is a candidate for top k, allocate a buffer position for it
        if (local_dist < radius)
        {
            buffer_pos = atomicAdd(&buffer_size, 1);
        }

        // merge the buffer if necessary
        for (;;)
        {
            // try to add the loaded value to the buffer (negative position indicates no value)
            if (0 <= buffer_pos && buffer_pos < k)
            {
                block_b[buffer_pos].distance = local_dist;
                block_b[buffer_pos].index = local_label;
            }

            // check for buffer overflow
            bool overflown = buffer_pos >= static_cast<std::int32_t>(k);
            // update buffer_pos for the next iteration
            buffer_pos -= k;

            // we don't have to merge if the buffer is not full
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            radius = merge_buffer<USE_WARP_SORT>(block_a, block_b, &buffer_size, k);
        }
    }

    radius = merge_buffer<USE_WARP_SORT>(block_a, block_b, &buffer_size, k);

    // copy the values to the output
    for (std::size_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        out_dist(blockIdx.x, i) = block_a[i].distance;
        out_label(blockIdx.x, i) = block_a[i].index;
    }
}

template <bool USE_WARP_SORT, bool PREFETCH_NEXT_BATCH, std::size_t K, std::size_t BLOCK_SIZE,
          std::size_t BATCH_SIZE>
__global__ void buffered_partial_bitonic_aos(array_view<float, 2> input,
                                             array_view<float, 2> out_dist,
                                             array_view<std::int32_t, 2> out_label)
{
    constexpr std::size_t PRESORTED_SIZE = USE_WARP_SORT ? 32 : 1;

    extern __shared__ std::uint8_t shm[];
    __shared__ std::int32_t buffer_size;

    // split shared memory to two blocks
    knn::pair_t* block_a = reinterpret_cast<knn::pair_t*>(shm);
    knn::pair_t* block_b = block_a + K;

    if (threadIdx.x == 0)
    {
        buffer_size = 0;
    }

    const auto input_count = input.size(1);

    // reset buffer
    for (std::size_t i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        block_b[i].distance = std::numeric_limits<float>::infinity();
    }

    // load and sort the first block
    load_block<USE_WARP_SORT, aos_layout>({block_a}, 0, input, K);
    block_sort<USE_WARP_SORT, PRESORTED_SIZE, K, BLOCK_SIZE, aos_layout>({block_a});

    // initialize the radius to the kth element
    auto radius = block_a[K - 1].distance;

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
            dist[j] = std::numeric_limits<float>::infinity();
            buffer_pos[j] = -1; // none (do not add this item to the buffer)
            label[j] = i + j * BLOCK_SIZE;

            // read the next value from input
            if (label[j] < input_count)
            {
                dist[j] = input(blockIdx.x, label[j]);
            }
        }

        // prefetch the next batch
        if (PREFETCH_NEXT_BATCH)
        {
#pragma unroll
            for (std::size_t j = 0; j < BATCH_SIZE; ++j)
            {
                const auto point_idx = i + BLOCK_SIZE * BATCH_SIZE + j * BLOCK_SIZE;
                if (point_idx < input_count)
                {
                    prefetch(input.ptr(blockIdx.x, point_idx));
                }
            }
        }

// allocate buffer positions if necessary
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
                    block_b[buffer_pos[j]].distance = dist[j];
                    block_b[buffer_pos[j]].index = label[j];
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

            radius = merge_buffer<USE_WARP_SORT, K, BLOCK_SIZE>(block_a, block_b, &buffer_size);
        }
    }

    radius = merge_buffer<USE_WARP_SORT, K, BLOCK_SIZE>(block_a, block_b, &buffer_size);

    // copy the values to the output
    for (std::size_t i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        out_dist(blockIdx.x, i) = block_a[i].distance;
        out_label(blockIdx.x, i) = block_a[i].index;
    }
}

} // namespace

void buffered_partial_bitonic::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();

    buffered_partial_bitonic_aos<true>
        <<<block_count, selection_block_size(),
           2 * k() * (sizeof(float) + sizeof(std::int32_t)) + sizeof(std::int32_t)>>>(
            in_dist_gpu(), out_dist_gpu(), out_label_gpu(), k());

    cuda_stream::make_default().sync();
}

void static_buffered_partial_bitonic::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();
    const auto shm_size = 2 * k() * (sizeof(float) + sizeof(std::int32_t));
    const auto block_size = selection_block_size();

    // use warp shuffles in bitonic sort
    constexpr bool USE_WARP_SORT = true;
    // prefetch the next batch of distances
    constexpr bool PREFETCH_NEXT_BATCH = false;

    if (!dynamic_switch<TOPK_SINGLEPASS_K_VALUES>(k(), [=]<std::size_t K>() {
            if (!dynamic_switch<128, 256, 512>(block_size, [=]<std::size_t BlockSize>() {
                    constexpr std::size_t BATCH_SIZE = 2;
                    buffered_partial_bitonic_aos<USE_WARP_SORT, PREFETCH_NEXT_BATCH, K, BlockSize,
                                                 BATCH_SIZE>
                        <<<block_count, BlockSize, shm_size>>>(dist, out_dist, out_label);
                }))
            {
                throw std::runtime_error("Unsupported block size: " + std::to_string(block_size));
            }
        }))
    {
        throw std::runtime_error("Unsupported k value: " + std::to_string(k()));
    }

    cuda_stream::make_default().sync();
}
