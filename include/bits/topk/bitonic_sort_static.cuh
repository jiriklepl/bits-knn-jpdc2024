#ifndef BITONIC_SORT_STATIC_CUH_
#define BITONIC_SORT_STATIC_CUH_

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "bits/topk/bitonic_sort_warp.cuh"

// block_sort_bitonic version with compile time constants instead of parameters
template <bool USE_WARP_SORT, std::size_t STRIDE, std::size_t VALUE_COUNT, std::size_t BLOCK_SIZE,
          typename Layout>
__device__ __forceinline__ void block_sort_bitonic(Layout values)
{
    // minimal stride handeled using shared memory
    constexpr std::size_t MIN_BLOCK_STRIDE = USE_WARP_SORT ? 16 : 0;

    const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
    for (std::size_t stride = STRIDE; stride > MIN_BLOCK_STRIDE; stride /= 2)
    {
        const auto mask = stride - 1;

        // small arrays will have some threads idle
        if (VALUE_COUNT / 2 >= BLOCK_SIZE || thread_idx < VALUE_COUNT / 2)
        {
// variable j iterates over compare and swap operations
#pragma unroll
            for (std::size_t j = 0; j < VALUE_COUNT / 2; j += BLOCK_SIZE)
            {
                // to map index of the operation j to index in the array i, we have to skip every
                // odd block of size `stride`
                const auto op_idx = j + thread_idx;
                const auto lhs_idx = ((op_idx & ~mask) * 2) + (op_idx & mask);
                const auto rhs_idx = lhs_idx ^ stride;
                assert(lhs_idx < rhs_idx);

                if (values.dist(lhs_idx) > values.dist(rhs_idx))
                {
                    values.swap(lhs_idx, rhs_idx);
                }
            }
        }

        __syncthreads();
    }

    if (USE_WARP_SORT)
    {
        // small arrays will have some threads idle
        if (VALUE_COUNT >= BLOCK_SIZE || thread_idx < VALUE_COUNT)
        {
// use warp shuffles for strides <= 16
#pragma unroll
            for (std::size_t i = 0; i < VALUE_COUNT; i += BLOCK_SIZE)
            {
                const auto value_idx = i + thread_idx;
                auto local_dist = values.dist(value_idx);
                auto local_label = values.label(value_idx);

                warp_sort_bitonic(local_dist, local_label, std::min<std::size_t>(STRIDE, 16));

                // store the sorted distances to shared memory
                values.dist(value_idx) = local_dist;
                values.label(value_idx) = local_label;
            }
        }

        __syncthreads();
    }
}

// block_merge version with compile time constants instead of parameters
template <bool USE_WARP_SORT, std::size_t STRIDE, std::size_t VALUE_COUNT, std::size_t BLOCK_SIZE,
          typename Layout>
__device__ __forceinline__ void block_merge(Layout values)
{
    const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (!USE_WARP_SORT || STRIDE >= 32)
    {
        const auto mask = STRIDE * 2 - 1;
        const auto half_mask = mask >> 1;

        if (VALUE_COUNT / 2 >= BLOCK_SIZE || thread_idx < VALUE_COUNT / 2)
        {
#pragma unroll
            for (std::size_t j = 0; j < VALUE_COUNT / 2; j += BLOCK_SIZE)
            {
                const auto op_idx = j + thread_idx;
                // skip odd blocks (assuming blocks of size SIZE / 2)
                const auto lhs_idx = ((op_idx & ~half_mask) * 2) + (op_idx & half_mask);
                const auto rhs_idx = lhs_idx ^ mask;
                assert(lhs_idx < rhs_idx);

                if (values.dist(lhs_idx) > values.dist(rhs_idx))
                {
                    values.swap(lhs_idx, rhs_idx);
                }
            }
        }
    }
    else // use warp shuffles for sizes <= 32
    {
        if (VALUE_COUNT >= BLOCK_SIZE || thread_idx < VALUE_COUNT)
        {
#pragma unroll
            for (std::size_t i = 0; i < VALUE_COUNT; i += BLOCK_SIZE)
            {
                const auto value_idx = i + thread_idx;
                auto local_dist = values.dist(value_idx);
                auto local_label = values.label(value_idx);

                warp_reversed_bitonic_stage(local_dist, local_label, STRIDE);

                // store the sorted distances to shared memory
                values.dist(value_idx) = local_dist;
                values.label(value_idx) = local_label;
            }
        }
    }
    __syncthreads();
}

template <bool USE_WARP_SORT, std::size_t STRIDE, std::size_t VALUE_COUNT, std::size_t BLOCK_SIZE,
          typename Layout>
__device__ void block_sort(Layout values)
{
    if (STRIDE >= VALUE_COUNT)
    {
        return;
    }

    block_merge<USE_WARP_SORT, STRIDE, VALUE_COUNT, BLOCK_SIZE, Layout>(values);
    block_sort_bitonic<USE_WARP_SORT, STRIDE / 2, VALUE_COUNT, BLOCK_SIZE, Layout>(values);

    block_sort<USE_WARP_SORT, STRIDE * 2, VALUE_COUNT, BLOCK_SIZE, Layout>(values);
}

#endif // BITONIC_SORT_STATIC_CUH_
