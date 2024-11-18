#ifndef BITS_TOPK_BITONIC_SORT_REGS_CUH_
#define BITS_TOPK_BITONIC_SORT_REGS_CUH_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <cooperative_groups.h>

#include "bits/memory.cuh"
#include "bits/topk/bitonic_sort_warp.cuh"

/** Reverse a register array
 *
 * @tparam T type of values in the array
 * @tparam SIZE size of the array
 *
 * @param data register array to reverse
 */
template <typename T, std::size_t SIZE>
__device__ __forceinline__ void reverse_regs(T (&data)[SIZE])
{
#pragma push
#pragma nv_diag_suppress = unsigned_compare_with_zero

#pragma unroll
    for (std::size_t i = 0; i < SIZE / 2; ++i)
    {
        swap_values(data[i], data[SIZE - 1 - i]);
    }

#pragma pop
}

/** Compare and swap two distance-label pairs.
 *
 * Update the input pairs so that `lhs` is less than or equal to `rhs`.
 *
 * @tparam ORDER ascending or descending order
 * @tparam Dist type of the distance values
 * @tparam Label type of the label values
 *
 * @param lhs left-hand side distance
 * @param lhs_label left-hand side label
 * @param rhs right-hand side distance
 * @param rhs_label right-hand side label
 */
template <order_t ORDER = order_t::ascending, typename Dist, typename Label>
__device__ __forceinline__ void compare_and_swap(Dist& lhs, Label& lhs_label, Dist& rhs,
                                                 Label& rhs_label)
{
    const bool swap_pred =
        (ORDER == order_t::ascending && lhs > rhs) || (ORDER == order_t::descending && lhs < rhs);

    if (swap_pred)
    {
        swap_values(lhs, rhs);
        swap_values(lhs_label, rhs_label);
    }
}

/** Single reversed Bitonic separator.
 *
 * Visualization for stride 4 and 8 values:
 *
 * <pre>
 * a0, a1, a2, a3, a4, a5, a6, a7
 *  |---------------------------|  <- compare-and-swap
 *      |-------------------|
 *          |-----------|
 *              |---|
 * </pre>
 *
 * @tparam STRIDE stride of the Bitonic separator (i.e., subsequences of size `STRIDE` are sorted)
 * @tparam BLOCK_SIZE number of threads in a thread block
 * @tparam VALUE_COUNT actual size of the register array (in case it is smaller than BLOCK_SIZE)
 * @tparam ITEMS_PER_THREAD number of values stored by each thread
 * @tparam ORDER ascending or descending order
 *
 * @param dist register array of distances
 * @param label register array of labels
 * @param shm_dist shared memory region for distances (size should be at least `2 * BLOCK_SIZE`)
 * @param shm_label shared memory region for labels (size should be at least `2 * BLOCK_SIZE`)
 */
template <std::size_t STRIDE, std::size_t BLOCK_SIZE, std::size_t VALUE_COUNT,
          std::size_t ITEMS_PER_THREAD, order_t ORDER = order_t::ascending>
__device__ void reversed_bitonic_stage(float (&dist)[ITEMS_PER_THREAD],
                                       std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                       std::int32_t* shm_label)
{
    // number of threads per warp
    constexpr std::size_t WARP_SIZE = 32;
    // every stride greater than or equal to this threshold will be done in shared memory
    constexpr std::size_t WARP_CAPACITY = WARP_SIZE * ITEMS_PER_THREAD;

#pragma push
#pragma nv_diag_suppress = unsigned_compare_with_zero

    if constexpr (STRIDE <= 0)
    {
        return; // end when we reach stride 0
    }

    const auto block = cooperative_groups::this_thread_block();
    const auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);

    // merge locally without any synchronization
    if constexpr (STRIDE < ITEMS_PER_THREAD)
    {
        constexpr std::size_t STRIDE_MASK = STRIDE - 1;
        constexpr std::size_t SUBSEQUENCE_MASK = STRIDE * 2 - 1;

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD / 2; ++i)
        {
            const auto lhs_idx = ((i & ~STRIDE_MASK) * 2) + (i & STRIDE_MASK);
            const auto rhs_idx = lhs_idx ^ SUBSEQUENCE_MASK;
            assert(lhs_idx < rhs_idx);

            compare_and_swap<ORDER>(dist[lhs_idx], label[lhs_idx], dist[rhs_idx], label[rhs_idx]);
        }
    }
    else if constexpr (/* ITEMS_PER_THREAD <= */ STRIDE < WARP_CAPACITY) // merge within warp
    {
        // stride in number of threads
        constexpr std::size_t WARP_STRIDE = STRIDE / ITEMS_PER_THREAD;

        if ((warp.thread_rank() & WARP_STRIDE) != 0)
        {
            reverse_regs(dist);
            reverse_regs(label);
        }

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            warp_reversed_bitonic_stage<float, std::int32_t, ORDER>(dist[i], label[i], WARP_STRIDE);
        }

        if ((warp.thread_rank() & WARP_STRIDE) != 0)
        {
            reverse_regs(dist);
            reverse_regs(label);
        }
    }
    else // merge in shared memory
    {
        // stride in number of threads
        constexpr std::size_t BLOCK_STRIDE = STRIDE / ITEMS_PER_THREAD;

        constexpr std::size_t STRIDE_MASK = BLOCK_STRIDE - 1;
        constexpr std::size_t SUBSEQUENCE_MASK = BLOCK_STRIDE * 2 - 1;

        if ((block.thread_rank() & BLOCK_STRIDE) != 0)
        {
            reverse_regs(dist);
            reverse_regs(label);
        }

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; i += 2)
        {
            if (ITEMS_PER_THREAD > 1 || BLOCK_SIZE <= VALUE_COUNT ||
                block.thread_rank() < VALUE_COUNT)
            {
                shm_dist[block.thread_rank()] = dist[i];
                shm_label[block.thread_rank()] = label[i];
            }

            if constexpr (ITEMS_PER_THREAD > 1)
            {
                shm_dist[BLOCK_SIZE + block.thread_rank()] = dist[i + 1];
                shm_label[BLOCK_SIZE + block.thread_rank()] = label[i + 1];
            }

            __syncthreads();

            constexpr std::size_t LAST_OP = std::min<std::size_t>(BLOCK_SIZE, VALUE_COUNT) / 2;
            if (ITEMS_PER_THREAD > 1 || block.thread_rank() < LAST_OP)
            {
                const auto lhs_idx = ((block.thread_rank() & ~STRIDE_MASK) * 2) +
                                     (block.thread_rank() & STRIDE_MASK);
                const auto rhs_idx = lhs_idx ^ SUBSEQUENCE_MASK;
                assert(lhs_idx < rhs_idx);

                compare_and_swap<ORDER>(shm_dist[lhs_idx], shm_label[lhs_idx], shm_dist[rhs_idx],
                                        shm_label[rhs_idx]);
            }

            __syncthreads();

            // load values back to registers
            if (ITEMS_PER_THREAD > 1 || BLOCK_SIZE <= VALUE_COUNT ||
                block.thread_rank() < VALUE_COUNT)
            {
                dist[i] = shm_dist[block.thread_rank()];
                label[i] = shm_label[block.thread_rank()];
            }

            if constexpr (ITEMS_PER_THREAD > 1)
            {
                dist[i + 1] = shm_dist[BLOCK_SIZE + block.thread_rank()];
                label[i + 1] = shm_label[BLOCK_SIZE + block.thread_rank()];
            }

            __syncthreads();
        }

        if ((block.thread_rank() & BLOCK_STRIDE) != 0)
        {
            reverse_regs(dist);
            reverse_regs(label);
        }
    }

#pragma pop
}

/** Sort array of bitonic subsequences of size `STRIDE`.
 *
 * Visualization of this operation for 8 values and stride 4:
 *
 * <pre>
 * r0, r1, r2, r3, r4, r5, r6, r7
 *  |---------------|               <- compare-and-swap
 *      |---------------|
 *          |---------------|
 *              |---------------|
 *  |-------|       |-------|
 *      |-------|       |-------|
 *  |---|   |---|   |---|   |---|
 * </pre>
 *
 * @tparam STRIDE stride of the bitonic sort (i.e., subsequences of size `STRIDE` are bitonic)
 * @tparam BLOCK_SIZE number of threads in a thread block
 * @tparam VALUE_COUNT actual size of the register array (in case it is smaller than BLOCK_SIZE)
 * @tparam ITEMS_PER_THREAD number of values stored by each thread
 * @tparam ORDER ascending or descending order
 *
 * @param dist register array of distances
 * @param label register array of labels
 * @param shm_dist shared memory region for distances (size should be at least `2 * BLOCK_SIZE`)
 * @param shm_label shared memory region for labels (size should be at least `2 * BLOCK_SIZE`)
 */
template <std::size_t STRIDE, std::size_t BLOCK_SIZE, std::size_t VALUE_COUNT,
          std::size_t ITEMS_PER_THREAD, order_t ORDER = order_t::ascending>
__device__ void block_sort_bitonic(float (&dist)[ITEMS_PER_THREAD],
                                   std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                   std::int32_t* shm_label)
{
    // nvcc does not like ITEMS_PER_THREAD / 2 in comparison if ITEMS_PER_THREAD == 1
#pragma push
#pragma nv_diag_suppress = unsigned_compare_with_zero

    // number of threads per warp
    constexpr std::size_t WARP_SIZE = 32;
    // every stride greater than or equal to this threshold will be done in shared memory
    constexpr std::size_t WARP_CAPACITY = WARP_SIZE * ITEMS_PER_THREAD;

    if constexpr (STRIDE <= 0)
    {
        return; // end when we reach stride 0
    }

    const auto block = cooperative_groups::this_thread_block();

    // bitonic sort in shared memory
    if constexpr (STRIDE >= WARP_CAPACITY)
    {
        // stride in number of threads
        constexpr std::size_t BLOCK_STRIDE = STRIDE / ITEMS_PER_THREAD;

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; i += 2)
        {
            // Store 2 pairs to shared memory (so that all threads have something to work on).
            // Threads store consecutive values. However, we load them to two separate blocks to
            // avoid bank conflicts.
            if (ITEMS_PER_THREAD > 1 || BLOCK_SIZE <= VALUE_COUNT ||
                block.thread_rank() < VALUE_COUNT)
            {
                shm_dist[block.thread_rank()] = dist[i];
                shm_label[block.thread_rank()] = label[i];
            }

            if (ITEMS_PER_THREAD > 1)
            {
                shm_dist[BLOCK_SIZE + block.thread_rank()] = dist[i + 1];
                shm_label[BLOCK_SIZE + block.thread_rank()] = label[i + 1];
            }

            __syncthreads();

// bitonic sort steps in shared memory
#pragma unroll
            for (std::size_t stride = BLOCK_STRIDE; stride >= WARP_SIZE; stride /= 2)
            {
                const auto mask = stride - 1;

                const auto lhs_idx =
                    ((block.thread_rank() & ~mask) * 2) + (block.thread_rank() & mask);
                const auto rhs_idx = lhs_idx ^ stride;
                assert(lhs_idx < rhs_idx);

                constexpr std::size_t LAST_OP = std::min<std::size_t>(BLOCK_SIZE, VALUE_COUNT) / 2;

                // half of all threads will be idle if we only have one item per thread
                if (ITEMS_PER_THREAD > 1 || block.thread_rank() < LAST_OP)
                {
                    compare_and_swap<ORDER>(shm_dist[lhs_idx], shm_label[lhs_idx],
                                            shm_dist[rhs_idx], shm_label[rhs_idx]);
                }

                __syncthreads();
            }

            // load values back to registers
            if (ITEMS_PER_THREAD > 1 || BLOCK_SIZE <= VALUE_COUNT ||
                block.thread_rank() < VALUE_COUNT)
            {
                dist[i] = shm_dist[block.thread_rank()];
                label[i] = shm_label[block.thread_rank()];
            }

            if constexpr (ITEMS_PER_THREAD > 1)
            {
                dist[i + 1] = shm_dist[BLOCK_SIZE + block.thread_rank()];
                label[i + 1] = shm_label[BLOCK_SIZE + block.thread_rank()];
            }

            __syncthreads();
        }
    }

    // use warp shuffles for strides within warps
    if constexpr (STRIDE >= ITEMS_PER_THREAD)
    {
        // stride in number of threads
        constexpr std::size_t WARP_STRIDE =
            std::min<std::size_t>(STRIDE / ITEMS_PER_THREAD, WARP_SIZE / 2);

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            warp_sort_bitonic<float, std::int32_t, ORDER>(dist[i], label[i], WARP_STRIDE);
        }
    }

    // small strides on thread local data
    constexpr std::size_t LOCAL_STRIDE = std::min<std::size_t>(STRIDE, ITEMS_PER_THREAD / 2);

#pragma unroll
    for (std::size_t stride = LOCAL_STRIDE; stride > 0; stride /= 2)
    {
        const auto mask = stride - 1;

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD / 2; ++i)
        {
            const auto lhs_idx = ((i & ~mask) * 2) + (i & mask);
            const auto rhs_idx = lhs_idx ^ stride;
            assert(lhs_idx < rhs_idx);

            compare_and_swap<ORDER>(dist[lhs_idx], label[lhs_idx], dist[rhs_idx], label[rhs_idx]);
        }
    }

#pragma pop
}

/** Auxiliary block_sort function in which the control variable (ITERATION) decreases to 0.
 *
 * @tparam ITERATION `VALUE_COUNT / the current stride length`.
 * @tparam END_STRIDE if the current stride length is >= this value, the function ends.
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam VALUE_COUNT number of values in the whole thread block array (in case it is smaller than
 * BLOCK_SIZE)
 * @tparam ITEMS_PER_THREAD number of items stored in each thread
 * @tparam ORDER ascending or descending order
 * @param dist thread-local array of distances stored by this thread
 * @param label thread-local array of labels stored by this thread
 * @param shm_dist shared memory for distances (size should be at least `2 * BLOCK_SIZE`)
 * @param shm_label shared memory for labels (size should be at least `2 * BLOCK_SIZE`)
 */
template <std::size_t ITERATION, std::size_t END_STRIDE, std::size_t BLOCK_SIZE,
          std::size_t VALUE_COUNT, std::size_t ITEMS_PER_THREAD, order_t ORDER = order_t::ascending>
__device__ inline void block_sort_aux(float (&dist)[ITEMS_PER_THREAD],
                                      std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                      std::int32_t* shm_label)
{
    constexpr std::size_t STRIDE = ITERATION > 0 ? VALUE_COUNT / ITERATION : VALUE_COUNT;

    if constexpr (STRIDE >= END_STRIDE)
    {
        return; // the array is already sorted
    }

    reversed_bitonic_stage<STRIDE, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, ORDER>(
        dist, label, shm_dist, shm_label);
    block_sort_bitonic<STRIDE / 2, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, ORDER>(
        dist, label, shm_dist, shm_label);

    // given array of sorted subsequences of size STRIDE * 2 sort subsequences of size STRIDE * 4
    block_sort_aux<ITERATION / 2, END_STRIDE, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, ORDER>(
        dist, label, shm_dist, shm_label);
}

/** Sort a block-wide register array of distance, label pairs using Bitonic sort.
 *
 * The output is in the blocked arrangement (i.e., each thread stores several consecutive values).
 *
 * @tparam STRIDE the current Bitonic sort stride length (use 1 if the input array is not sorted at
 * all).
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam VALUE_COUNT number of values in the whole thread block array (in case it is smaller than
 * BLOCK_SIZE)
 * @tparam ITEMS_PER_THREAD number of items stored in each thread
 * @tparam ORDER ascending or descending order
 * @param dist thread-local array of distances stored by this thread
 * @param label thread-local array of labels stored by this thread
 * @param shm_dist shared memory for distances (size should be at least `2 * BLOCK_SIZE`)
 * @param shm_label shared memory for labels (size should be at least `2 * BLOCK_SIZE`)
 */
template <std::size_t STRIDE, std::size_t BLOCK_SIZE, std::size_t VALUE_COUNT,
          std::size_t ITEMS_PER_THREAD, order_t ORDER = order_t::ascending>
__device__ inline void block_sort(float (&dist)[ITEMS_PER_THREAD],
                                  std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                  std::int32_t* shm_label)
{
    block_sort_aux<VALUE_COUNT / STRIDE, VALUE_COUNT, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD,
                   ORDER>(dist, label, shm_dist, shm_label);
}

/** Sort multiple small arrays in parallel using a block-wide Bitonic sort.
 *
 * The output is in the blocked arrangement (i.e., each thread stores several consecutive values).
 *
 * @tparam STRIDE the current Bitonic sort stride length (use 1 if the input array is not sorted at
 * all)
 * @tparam END_STRIDE size of the small arrays.
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam VALUE_COUNT number of values in the whole thread block array (in case it is smaller than
 * BLOCK_SIZE)
 * @tparam ITEMS_PER_THREAD number of items stored in each thread
 * @tparam ORDER ascending or descending order
 * @param dist thread-local array of distances stored by this thread
 * @param label thread-local array of labels stored by this thread
 * @param shm_dist shared memory for distances (size should be at least `2 * BLOCK_SIZE`)
 * @param shm_label shared memory for labels (size should be at least `2 * BLOCK_SIZE`)
 */
template <std::size_t STRIDE, std::size_t END_STRIDE, std::size_t BLOCK_SIZE,
          std::size_t VALUE_COUNT, std::size_t ITEMS_PER_THREAD, order_t ORDER = order_t::ascending>
__device__ inline void block_sort_partial(float (&dist)[ITEMS_PER_THREAD],
                                          std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                          std::int32_t* shm_label)
{
    block_sort_aux<VALUE_COUNT / STRIDE, END_STRIDE, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD,
                   ORDER>(dist, label, shm_dist, shm_label);
}

#endif // BITS_TOPK_BITONIC_SORT_REGS_CUH_
