#ifndef BITONIC_SORT_GLOBAL_CUH_
#define BITONIC_SORT_GLOBAL_CUH_

#include <cassert>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

#include "bits/memory.cuh"

#include "bits/topk/bitonic_sort_regs.cuh"
#include "bits/topk/bitonic_sort_warp.cuh"

/** Implementation of algorithms based on bitonic sort
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers for elements per thread
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD>
struct bitonic_sorter
{
    using key_t = float;
    using value_t = std::int32_t;

    // auxiliary shared memory for keys
    key_t* shm_key;
    // auxiliary shared memory for values
    value_t* shm_value;

    // number of elements that can be stored in registers/shared memory of a thread block
    static constexpr std::size_t BUFFER_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    __device__ bitonic_sorter(key_t* shm_key, value_t* shm_value)
        : shm_key(shm_key), shm_value(shm_value)
    {
    }

    /** Merge sorted subsequences of size @p stride into sorted subsequences of size @p stride * 2
     *
     * This operation works in-place on data in global memory ( @p global_key and @p global_value )
     *
     * Note, @p stride should be greater than or equal to @p BLOCK_SIZE * @p ITEMS_PER_THREAD
     * (i.e., this function should only be used to merge blocks that do not fit in registers of
     * a single thread block)
     *
     * @tparam ORDER order of the result (ascending or descending)
     *
     * @param key Registers to hold keys
     * @param value Registers to hold values
     * @param global_key Global memory with keys
     * @param global_value Global memory with values
     * @param value_count Total number of elements in global memory
     * @param stride The largest stride performed by this function. All consecutive subsequences of
     *               size @p stride in global memory have to be sorted. This function will merge
     *               consecutive subsequences of size @p stride into sorted subsequences of size
     *               @p stride * 2
     */
    template <order_t ORDER = order_t::ascending>
    __device__ __forceinline__ void
    merge(key_t (&key)[ITEMS_PER_THREAD], value_t (&value)[ITEMS_PER_THREAD], key_t* global_key,
          value_t* global_value, std::size_t value_count, std::size_t stride)
    {
        reversed_bitonic_stage<ORDER>(global_key, global_value, value_count, stride);
        sort_bitonic<ORDER>(key, value, global_key, global_value, value_count, stride / 2);
    }

    /** Execute the reversed bitonic stage in global memory.
     *
     * Visualization of this operation for 8 values and stride 4:
     *
     * <pre>
     * a0, a1, a2, a3, a4, a5, a6, a7
     *  |---------------------------|  <- compare-and-swap
     *      |-------------------|
     *          |-----------|
     *              |---|
     * </pre>
     *
     * @tparam ORDER order of the result (ascending or descending)
     *
     * @param global_key Global memory with keys
     * @param global_value Global memory with values
     * @param value_count Total number of elements in global memory
     * @param stride Stride of this operation. Subsequences of size @p stride have to be sorted.
     *               This operation will make subsequences of size @p stride * 2 bitonic.
     */
    template <order_t ORDER = order_t::ascending>
    __device__ __forceinline__ void reversed_bitonic_stage(key_t* global_key, value_t* global_value,
                                                           std::size_t value_count,
                                                           std::size_t stride)
    {
        // linearized thread ID within the thread block
        const auto thread_idx = threadIdx.x + blockDim.x * threadIdx.y;

        // execute the first reversed bitonic stage in global memory
        const auto stride_mask = stride - 1;
        const auto subsequence_mask = stride * 2 - 1;

        // for all compare and swap operations
        for (std::size_t i = thread_idx; i < value_count / 2; i += BLOCK_SIZE)
        {
            // compute data indices of both operands to compare-and-swap
            const auto lhs_idx = ((i & ~stride_mask) * 2) + (i & stride_mask);
            const auto rhs_idx = lhs_idx ^ subsequence_mask;
            assert(lhs_idx < rhs_idx);

            const bool swap =
                (ORDER == order_t::ascending && global_key[lhs_idx] > global_key[rhs_idx]) ||
                (ORDER == order_t::descending && global_key[lhs_idx] < global_key[rhs_idx]);

            if (swap)
            {
                swap_values(global_key[lhs_idx], global_key[rhs_idx]);
                swap_values(global_value[lhs_idx], global_value[rhs_idx]);
            }
        }

        __syncthreads();
    }

    /** Execute bitonic sort steps locally in registers of each thread.
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
     * @tparam ORDER order of the sorted list
     * @tparam STRIDE the largest stride used by this function
     * @tparam SIZE number of values in the register array @p key and @p value
     *
     * @param key List of keys
     * @param value List of values
     */
    template <order_t ORDER, std::size_t STRIDE, std::size_t SIZE>
    __device__ __forceinline__ void sort_bitonic_local(key_t (&key)[SIZE], value_t (&value)[SIZE])
    {
#pragma unroll
        for (std::size_t stride = STRIDE; stride > 0; stride /= 2)
        {
            const auto mask = stride - 1;

#pragma unroll
            for (std::size_t i = 0; i < SIZE / 2; ++i)
            {
                const auto lhs_idx = ((i & ~mask) * 2) + (i & mask);
                const auto rhs_idx = lhs_idx ^ stride;
                assert(rhs_idx > lhs_idx);

                const bool swap = (ORDER == order_t::ascending && key[lhs_idx] > key[rhs_idx]) ||
                                  (ORDER == order_t::descending && key[lhs_idx] < key[rhs_idx]);

                if (swap)
                {
                    swap_values(key[lhs_idx], key[rhs_idx]);
                    swap_values(value[lhs_idx], value[rhs_idx]);
                }
            }
        }
    }

    /** Execute bitonic sort steps in registers and shared memory across the whole thread block.
     *
     * Visualization of this operation for 8 values and stride 4 (each thread stores
     * a pair of values in parentheses):
     *
     * <pre>
     * (r0, r1), (r2, r3), (r4, r5), (r6, r7)
     *   |-------------------|               <- compare-and-swap
     *       |-------------------|
     *             |-------------------|
     *                 |-------------------|
     *   |---------|         |---------|
     *       |---------|         |---------|
     *   |---|     |---|     |---|     |---|
     * </pre>
     *
     * @tparam ORDER order of the sorted list
     * @tparam STRIDE the largest stride used by this function
     * @tparam SIZE number of values in the register array @p key and @p value
     *
     * @param key List of subsequent keys
     * @param value List of subsequent values
     */
    template <order_t ORDER, std::size_t STRIDE, std::size_t SIZE>
    __device__ __forceinline__ void sort_bitonic(key_t (&key)[SIZE], value_t (&value)[SIZE])
    {
        block_sort_bitonic<STRIDE, BLOCK_SIZE, BUFFER_SIZE, SIZE, ORDER>(key, value, shm_key,
                                                                         shm_value);
    }

    /** Sort bitonic subsequences of size @p stride * 2
     *
     * This operation works in global memory ( @p global_key and @p global_value )
     * and it is in-place.
     *
     * Visualization of this operation for 8 values and stride 4:
     *
     * <pre>
     * a0, a1, a2, a3, a4, a5, a6, a7
     *  |---------------|               <- compare-and-swap
     *      |---------------|
     *          |---------------|
     *              |---------------|
     *  |-------|       |-------|
     *      |-------|       |-------|
     *  |---|   |---|   |---|   |---|
     * </pre>
     *
     * @tparam ORDER order of the sorted result
     * @tparam COMBINED number of compare-and-swap operations in global memory performed in one
     * step. Combined operations use more registers but they reduce the number of global memory
     * transactions.
     *
     * @param key Registers to hold keys
     * @param value Registers to hold values
     * @param global_key Global memory with keys
     * @param global_value Global memory with values
     * @param value_count Total number of elements in global memory
     * @param stride The largest bitonic sort stride. Subsequences of size @p stride * 2 have to be
     * bitonic. This function will sort these subsequences.
     */
    template <order_t ORDER = order_t::ascending, std::size_t COMBINED = 2>
    __device__ __forceinline__ void sort_bitonic(key_t (&key)[ITEMS_PER_THREAD],
                                                 value_t (&value)[ITEMS_PER_THREAD],
                                                 key_t* global_key, value_t* global_value,
                                                 std::uint32_t value_count, std::size_t stride)
    {
        // linearized thread ID within the thread block
        const auto thread_idx = threadIdx.x + blockDim.x * threadIdx.y;

        // every stride greater then or equal to this threashold will be done in global memory
        constexpr std::size_t BLOCK_STRIDE_THRESHOLD = BLOCK_SIZE * ITEMS_PER_THREAD;
        // number of registers to cache values from global memory for a bitonic step
        constexpr std::size_t REG_COUNT = 1 << COMBINED;

        // execute bitonic steps with large stride in global memory
        for (; stride >= BLOCK_STRIDE_THRESHOLD; stride /= REG_COUNT)
        {
            // the smallest stride used in this iteration
            const auto small_stride = 2 * stride / REG_COUNT;
            const auto small_stride_mask = small_stride - 1;

            for (std::size_t i = thread_idx; i < value_count / REG_COUNT; i += BLOCK_SIZE)
            {
                // find the first index in global memory fetched by this thread
                const auto base_idx =
                    (i & ~small_stride_mask) * REG_COUNT + (i & small_stride_mask);

                // prepare auxiliary registers to hold keys/values
                key_t aux_key[REG_COUNT];
                value_t aux_value[REG_COUNT];

// load values to registers
#pragma unroll
                for (std::size_t j = 0; j < REG_COUNT; ++j)
                {
                    const auto global_idx = base_idx + j * small_stride;
                    aux_key[j] = global_key[global_idx];
                    aux_value[j] = global_value[global_idx];
                }

                // perform bitonic steps in registers
                sort_bitonic_local<ORDER, REG_COUNT / 2>(aux_key, aux_value);

// store the values to global memory
#pragma unroll
                for (std::size_t j = 0; j < REG_COUNT; ++j)
                {
                    const auto global_idx = base_idx + j * small_stride;
                    global_key[global_idx] = aux_key[j];
                    global_value[global_idx] = aux_value[j];
                }
            }
        }

        // execute bitonic steps with small strides in shared memory/registers
        constexpr std::size_t STRIDE = BLOCK_STRIDE_THRESHOLD / 2;

        for (std::size_t i = 0; i < value_count; i += BUFFER_SIZE)
        {
            // load BUFFER_SIZE values to registers
            load_blocked<BLOCK_SIZE>(key, global_key + i, shm_key);
            load_blocked<BLOCK_SIZE>(value, global_value + i, shm_value);

            block_sort_bitonic<STRIDE, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD, ORDER>(
                key, value, shm_key, shm_value);

            // store the values back to global memory
            store_blocked<BLOCK_SIZE>(key, global_key + i, shm_key);
            store_blocked<BLOCK_SIZE>(value, global_value + i, shm_value);
        }
    }

    /** Sort values in registers
     *
     * @tparam ORDER order of the result (ascending or descending)
     *
     * @param key Registers to with keys (in blocked arrangement)
     * @param value Registers to with values (in blocked arrangement)
     */
    template <order_t ORDER = order_t::ascending>
    __device__ __forceinline__ void sort(key_t (&key)[ITEMS_PER_THREAD],
                                         value_t (&value)[ITEMS_PER_THREAD])
    {
        block_sort<1, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD, ORDER>(key, value, shm_key,
                                                                        shm_value);
    }

    /** Sort values in global memory
     *
     * @param key Registers to hold keys
     * @param value Registers to hold values
     * @param global_key Global memory with keys
     * @param global_value Global memory with values
     * @param value_count Total number of elements in global memory
     */
    template <order_t ORDER = order_t::ascending>
    __device__ __forceinline__ void sort(key_t (&key)[ITEMS_PER_THREAD],
                                         value_t (&value)[ITEMS_PER_THREAD], key_t* global_key,
                                         value_t* global_value, std::size_t value_count)
    {
        // sort blocks that fit into registers/shared memory of a thread block
        for (std::size_t i = 0; i < value_count; i += BUFFER_SIZE)
        {
            // load BUFFER_SIZE values to registers
            load_blocked<BLOCK_SIZE>(key, global_key + i, shm_key);
            load_blocked<BLOCK_SIZE>(value, global_value + i, shm_value);

            // sort values in registers
            sort<ORDER>(key, value);

            // store the values back to global memory
            store_blocked<BLOCK_SIZE>(key, global_key + i, shm_key);
            store_blocked<BLOCK_SIZE>(value, global_value + i, shm_value);
        }

        // merge larger blocks
        for (std::size_t stride = BUFFER_SIZE; stride < value_count; stride *= 2)
        {
            merge<ORDER>(key, value, global_key, global_value, value_count, stride);
        }
    }
};

#endif // BITONIC_SORT_GLOBAL_CUH_
