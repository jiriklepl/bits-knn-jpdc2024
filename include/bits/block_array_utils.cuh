#ifndef BITS_BLOCK_ARRAY_UTILS_CUH_
#define BITS_BLOCK_ARRAY_UTILS_CUH_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/** Find lower bound for @p value in a sorted block-wide register array @p values
 *
 * Lower bound is the largest value in @p values < @p value
 *
 * @tparam BLOCK_SIZE Number of threads in each thread block
 * @tparam ITEMS_PER_THREAD Number of values per thread
 * @tparam T Data type
 *
 * @param value Searched value
 * @param values Block-wide register array where each thread stores consecutive values (i.e.,
                 blocked layout in CUB terminology)
 * @param value_count Number of valid values in the whole block-wide register array
 * @param shm auxiliary shared memory
 * @return index of the lower bound for @p value or -1 if all values are >= @p value
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
__device__ __forceinline__ std::int32_t lower_bound(T value, T (&values)[ITEMS_PER_THREAD],
                                                    std::size_t value_count, std::int32_t* shm)
{
    if (threadIdx.x == 0)
    {
        *shm = 0;
    }

    __syncthreads();

    std::int32_t count = 0;

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (idx < value_count && values[i] < value)
        {
            ++count;
        }
    }

    atomicAdd(shm, count);

    __syncthreads();

    count = *shm;

    __syncthreads();

    return count - 1;
}

/** Get value at a specified index in a block-wide register array stored in a blocked arrangement
 *
 * @tparam BLOCK_SIZE number of threads in a thread block
 * @tparam ITEMS_PER_THREAD number of items per thread in the array
 * @tparam T type of elements
 *
 * @param values Consecutive values from the array
 * @param query_idx Searched index
 * @param shm Shared memory for the operation
 * @return value at @p query_idx in a block-wide register array @p values
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
__device__ __forceinline__ T value_at(T (&values)[ITEMS_PER_THREAD], std::int32_t query_idx, T* shm)
{
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (query_idx == idx)
        {
            *shm = values[i];
        }
    }

    __syncthreads();

    const auto value = *shm;

    __syncthreads();

    return value;
}

#endif // BITS_BLOCK_ARRAY_UTILS_CUH_
