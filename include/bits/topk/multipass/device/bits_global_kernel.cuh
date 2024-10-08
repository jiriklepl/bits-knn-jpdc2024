#ifndef BITS_GLOBAL_KERNEL_CUH_
#define BITS_GLOBAL_KERNEL_CUH_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/knn.hpp"
#include "bits/topk/multipass/bits_global_kernel.hpp"

#include "bits/memory.cuh"
#include "bits/topk/bitonic_sort.cuh"
#include "bits/topk/bitonic_sort_regs.cuh"

template <std::size_t BLOCK_SIZE, std::size_t BUFFER_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void load_block(float (&dist)[ITEMS_PER_THREAD],
                                           std::int32_t (&label)[ITEMS_PER_THREAD],
                                           array_view<float, 2> input, std::size_t point_offset)
{
    const auto input_count = input.size(1);

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // load value from global memory
        label[i] = point_offset + i * BLOCK_SIZE + threadIdx.x;
        dist[i] = label[i] >= input_count ? std::numeric_limits<float>::infinity()
                                          : input(blockIdx.x, label[i]);
    }
}

template <std::size_t BLOCK_SIZE, std::size_t BUFFER_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
load_block(float (&dist)[ITEMS_PER_THREAD], std::int32_t (&label)[ITEMS_PER_THREAD],
           array_view<float, 2> global_dist, array_view<std::int32_t, 2> global_label,
           std::size_t point_offset)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // load value from global memory
        const auto idx = point_offset + i * BLOCK_SIZE + threadIdx.x;

        label[i] = global_label(blockIdx.x, idx);
        dist[i] = global_dist(blockIdx.x, idx);
    }
}

template <std::size_t BLOCK_SIZE, std::size_t BUFFER_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
store_block(array_view<float, 2> global_dist, array_view<std::int32_t, 2> global_label,
            float (&dist)[ITEMS_PER_THREAD], std::int32_t (&label)[ITEMS_PER_THREAD],
            std::size_t output_offset)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // load value from global memory
        const auto idx = output_offset + i * BLOCK_SIZE + threadIdx.x;

        global_label(blockIdx.x, idx) = label[i];
        global_dist(blockIdx.x, idx) = dist[i];
    }
}

template <std::size_t BUFFER_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ float broadcast_radius(float (&dist)[ITEMS_PER_THREAD], float* shm)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto element_idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (element_idx + 1 == BUFFER_SIZE)
        {
            *shm = dist[i];
        }
    }

    __syncthreads();

    const auto radius = *shm;

    __syncthreads();

    return radius;
}

template <std::size_t BLOCK_SIZE, std::size_t BUFFER_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ float
merge_buffer(float (&dist)[ITEMS_PER_THREAD], std::int32_t (&label)[ITEMS_PER_THREAD],
             float* shm_dist, std::int32_t* shm_label, array_view<float, 2> global_dist,
             array_view<std::int32_t, 2> global_label, std::int32_t* buffer_size, std::size_t k)
{
    float radius = -1;

    float buffer_dist[ITEMS_PER_THREAD];
    std::int32_t buffer_label[ITEMS_PER_THREAD];

// load buffer to registers
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto shm_idx = i * BLOCK_SIZE + threadIdx.x;
        buffer_dist[i] = shm_dist[shm_idx];
        buffer_label[i] = shm_label[shm_idx];
    }

    __syncthreads();

    // sort the buffer in descending order
    block_sort<1, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD, order_t::descending>(
        buffer_dist, buffer_label, shm_dist, shm_label);

    for (std::size_t i = 0; i < k; i += BUFFER_SIZE)
    {
        // loop invariant: buffer is sorted in descending order

        // load the block starting at i to registers
        load_block<BLOCK_SIZE, BUFFER_SIZE>(dist, label, global_dist, global_label, i);

// merge buffer with the block
#pragma unroll
        for (std::size_t j = 0; j < ITEMS_PER_THREAD; ++j)
        {
            if (dist[j] > buffer_dist[j])
            {
                swap_values(dist[j], buffer_dist[j]);
                swap_values(label[j], buffer_label[j]);
            }
        }

        // sort the block and the buffer
        block_sort_bitonic<BUFFER_SIZE / 2, BLOCK_SIZE, BUFFER_SIZE>(dist, label, shm_dist,
                                                                     shm_label);
        block_sort_bitonic<BUFFER_SIZE / 2, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD,
                           order_t::descending>(buffer_dist, buffer_label, shm_dist, shm_label);

        // update radius
        radius = std::max<float>(radius, broadcast_radius<BUFFER_SIZE>(dist, shm_dist));

        // store the block back to global memory
        store_block<BLOCK_SIZE, BUFFER_SIZE>(global_dist, global_label, dist, label, i);
    }

    // update buffer size
    if (threadIdx.x == 0)
    {
        *buffer_size =
            std::max<std::int32_t>(*buffer_size - static_cast<std::int32_t>(BUFFER_SIZE), 0);
    }

// reset buffer
#pragma unroll
    for (std::size_t i = 0; i < BUFFER_SIZE; i += BLOCK_SIZE)
    {
        const auto buffer_idx = i + threadIdx.x;
        if (buffer_idx < BUFFER_SIZE)
        {
            shm_dist[buffer_idx] = std::numeric_limits<float>::infinity();
        }
    }

    __syncthreads();

    return radius;
}

template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
__global__ void bits_global_kernel(array_view<float, 2> input, array_view<float, 2> out_dist,
                                   array_view<std::int32_t, 2> out_label, std::size_t k)
{
    constexpr std::size_t ITEMS_PER_THREAD = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    extern __shared__ std::uint8_t shm[];
    // number of elements in the buffer
    __shared__ std::int32_t shm_buffer_size;
    // buffer and memory for sorting
    float* shm_dist = reinterpret_cast<float*>(shm);
    std::int32_t* shm_label = reinterpret_cast<std::int32_t*>(shm_dist + BUFFER_SIZE);

    const auto input_count = input.size(1);
    // initialize the radius
    float radius = -1;

    // sort first k elements
    float topk_dist[ITEMS_PER_THREAD];
    std::int32_t topk_label[ITEMS_PER_THREAD];

    for (std::size_t i = 0; i < k; i += BUFFER_SIZE)
    {
        load_block<BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD>(topk_dist, topk_label, input, i);
        // sort the values
        block_sort<1, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD>(topk_dist, topk_label, shm_dist,
                                                                 shm_label);
        // broadcast the largest value and update radius
        radius = std::max<float>(radius, broadcast_radius<BUFFER_SIZE>(topk_dist, shm_dist));
        // store the values to global memory
        store_block<BLOCK_SIZE, BUFFER_SIZE>(out_dist, out_label, topk_dist, topk_label, i);
    }

    // initialize buffer size
    if (threadIdx.x == 0)
    {
        shm_buffer_size = 0;
    }

    // reset buffer
    for (std::size_t i = threadIdx.x; i < BUFFER_SIZE; i += BLOCK_SIZE)
    {
        shm_dist[i] = std::numeric_limits<float>::infinity();
    }

    __syncthreads();

    float dist[BATCH_SIZE];
    std::int32_t label[BATCH_SIZE];

    // while at least one thread in the block will read a value in the next iteration
    for (auto i = k + threadIdx.x; i < input_count + threadIdx.x; i += BLOCK_SIZE * BATCH_SIZE)
    {
        std::int32_t buffer_pos[BATCH_SIZE];

// load the next batch of distances
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            dist[j] = std::numeric_limits<float>::infinity();
            label[j] = i + j * BLOCK_SIZE;
            buffer_pos[j] = -1; // none (do not add this item to the buffer)

            // read the next value from input
            if (label[j] < input_count)
            {
                dist[j] = input(blockIdx.x, label[j]);
            }
        }

// allocate buffer positions
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            // if this pair is a candidate for top k, allocate a buffer position for it
            if (dist[j] < radius)
            {
                buffer_pos[j] = atomicAdd(&shm_buffer_size, 1);
            }
        }

        // merge the buffer
        for (;;)
        {
            bool overflown = false;

#pragma unroll
            for (std::size_t j = 0; j < BATCH_SIZE; ++j)
            {
                // try to add the loaded value to the buffer (negative position indicates no value)
                if (0 <= buffer_pos[j] && buffer_pos[j] < BUFFER_SIZE)
                {
                    shm_dist[buffer_pos[j]] = dist[j];
                    shm_label[buffer_pos[j]] = label[j];
                }

                // check for buffer overflow
                overflown |= buffer_pos[j] >= static_cast<std::int32_t>(BUFFER_SIZE);
                // update buffer position for the next iteration
                buffer_pos[j] -= BUFFER_SIZE;
            }

            // we don't have to merge if the buffer is not full
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            radius =
                merge_buffer<BLOCK_SIZE, BUFFER_SIZE>(topk_dist, topk_label, shm_dist, shm_label,
                                                      out_dist, out_label, &shm_buffer_size, k);
        }
    }

    radius = merge_buffer<BLOCK_SIZE, BUFFER_SIZE>(topk_dist, topk_label, shm_dist, shm_label,
                                                   out_dist, out_label, &shm_buffer_size, k);
}

#define DECL_BITS_GLOBAL_KERNEL(buffer_size, block_size, batch_size)                               \
    template void run_bits_global_kernel<buffer_size, block_size, batch_size>(                     \
        array_view<float, 2>, array_view<float, 2>, array_view<std::int32_t, 2>, std::size_t)

template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
void run_bits_global_kernel(array_view<float, 2> dist, array_view<float, 2> out_dist,
                            array_view<std::int32_t, 2> out_label, std::size_t k)
{
    const auto shm_size = BUFFER_SIZE * (sizeof(float) + sizeof(std::int32_t));
    const auto block_count = dist.size(0);

    bits_global_kernel<BUFFER_SIZE, BLOCK_SIZE, BATCH_SIZE>
        <<<block_count, BLOCK_SIZE, shm_size>>>(dist, out_dist, out_label, k);
}

#endif // BITS_GLOBAL_KERNEL_CUH_
