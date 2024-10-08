#ifndef TRANSPOSE_CUH_
#define TRANSPOSE_CUH_

#include <cstdio>

#include <cuda_runtime.h>

#include "bits/memory.cuh"

/** Rotate values in a register array @p values by one index to the left.
 *
 * @tparam T type of the values.
 * @tparam ITEMS_PER_THREAD size of the array.
 */
template <typename T, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void rotate_left(T (&values)[ITEMS_PER_THREAD])
{
    const auto first_value = values[0];

#pragma unroll
    for (std::size_t i = 0; i + 1 < ITEMS_PER_THREAD; ++i)
    {
        values[i] = values[i + 1];
    }

    values[ITEMS_PER_THREAD - 1] = first_value;
}

/** Rotate values in a register array @p values by @p shift
 *
 * @tparam T type of the values.
 * @tparam ITEMS_PER_THREAD size of the array.
 * @param shift stride length
 */
template <typename T, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void rotate_left(T (&values)[ITEMS_PER_THREAD], std::size_t shift)
{
    for (std::size_t i = 0; i < shift; ++i)
    {
        rotate_left(values);
    }
}

/** In-palce transpose of register arrays within each warp.
 *
 * The method is due to Catanzaro et al. - "A Decomposition for In-place Matrix Transposition"
 *
 * @tparam T type of stored values
 * @tparam ITEMS_PER_THREAD number of items stored in each thread (we assume warp size is divisible
           by this value)
 *
 * @param values Thread register array
 */
template <typename T, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ void transpose_warp(T (&values)[ITEMS_PER_THREAD])
{
    constexpr std::size_t WARP_SIZE = 32;
    constexpr std::size_t WHOLE_WARP = 0xFFFFFFFF;
    constexpr std::size_t PERIOD = WARP_SIZE / ITEMS_PER_THREAD;

    const auto thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const auto lane_id = thread_idx % WARP_SIZE;

    // rotate each thread array
    rotate_left(values, lane_id / PERIOD);

// row shuffle
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // compute the inverse of the d_i(j) from the paper
        auto target = lane_id + i * (WARP_SIZE - 1);
        if (i > (lane_id % ITEMS_PER_THREAD))
        {
            target += ITEMS_PER_THREAD;
        }
        target = ((target / ITEMS_PER_THREAD) % PERIOD) + (target % ITEMS_PER_THREAD) * PERIOD;

        values[i] = __shfl_sync(WHOLE_WARP, values[i], target);
    }

    // thread array shuffle (rotate by lane_id + shuffle which does not depend on lane_id)
    rotate_left(values, lane_id % ITEMS_PER_THREAD);

#pragma unroll
    for (std::size_t i = 1; i < ITEMS_PER_THREAD / 2; ++i)
    {
        swap_values(values[i], values[ITEMS_PER_THREAD - i]);
    }
}

#endif // TRANSPOSE_CUH_
