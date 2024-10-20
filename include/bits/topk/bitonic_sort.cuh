#ifndef BITONIC_SORT_CUH_
#define BITONIC_SORT_CUH_

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "bits/topk/bitonic_sort_layout.cuh"
#include "bits/topk/bitonic_sort_warp.cuh"

/** Naive implementation of Bitonic separator in shared memory.
 *
 * @tparam ORDER ascending or descending order.
 * @tparam Layout memory layout (structure-of-arrays or arrays-of-structures)
 * @param values values to sort
 * @param stride Bitonic sort stride length
 * @param values_count total number of values
 */
template <order_t ORDER, typename Layout>
__device__ __forceinline__ void bitonic_step(Layout values, std::uint32_t stride,
                                             std::uint32_t values_count)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto mask = stride - 1;

    // variable j iterates over compare and swap operations
    for (std::uint32_t j = block.thread_rank(); j < values_count / 2; j += block.size())
    {
        // to map index of the operation j to index in the array i, we have to skip every odd
        // block of size `stride`
        const auto i = ((j & ~mask) * 2) + (j & mask);
        const auto target = i ^ stride;
        assert(target > i);

        const bool swap = ORDER == order_t::ascending ? values.dist(i) > values.dist(target)
                                                      : values.dist(i) < values.dist(target);

        if (swap)
        {
            values.swap(i, target);
        }
    }

    __syncthreads();
}

/** Sort bitonic subsequences of size `stride / 2`.
 *
 * @tparam USE_WARP_SORT if true, use warp shuffles for small strides (<= 16).
 * @tparam ORDER ascending or descending order.
 * @tparam Layout memory layout (structure-of-arrays or arrays-of-structures).
 * @param values List of bitonic subsequences to be sorted
 * @param stride Size of the bitonic subsequences
 * @param values_count Total number of elements in the @p values list
 */
template <bool USE_WARP_SORT, order_t ORDER, typename Layout>
__device__ __forceinline__ void block_sort_bitonic(Layout values, std::uint32_t stride,
                                                   std::uint32_t values_count)
{
    constexpr std::uint32_t MIN_BLOCK_STRIDE = USE_WARP_SORT ? 16 : 0;

    for (; stride > MIN_BLOCK_STRIDE; stride /= 2)
    {
        bitonic_step<ORDER>(values, stride, values_count);
    }

    if (USE_WARP_SORT)
    {
        const auto block = cooperative_groups::this_thread_block();

        // use warp shuffles for strides <= 16
        for (std::uint32_t i = block.thread_rank(); i < values_count; i += block.size())
        {
            auto local_dist = values.dist(i);
            auto local_label = values.label(i);

            warp_sort_bitonic<float, std::int32_t, ORDER>(local_dist, local_label,
                                                          std::min<std::uint32_t>(stride, 16));

            // store the sorted distances to shared memory
            values.dist(i) = local_dist;
            values.label(i) = local_label;
        }

        __syncthreads();
    }
}

/** Reversed Bitonic separator.
 *
 * @tparam USE_WARP_SORT if true, use warp shuffles for small strides (<= 16).
 * @tparam ORDER ascending or descending order.
 * @tparam Layout memory layout (structure-of-arrays or arrays-of-structures).
 * @param values List of sorted subsequences of size `stride / 2`
 * @param values_count Total number of elements in the @p values list
 * @param stride size of sorted subsequences
 */
template <bool USE_WARP_SORT, order_t ORDER, typename Layout>
__device__ __forceinline__ void block_merge(Layout values, std::uint32_t stride,
                                            std::uint32_t values_count)
{
    const auto block = cooperative_groups::this_thread_block();

    if (!USE_WARP_SORT || stride >= 32)
    {
        const auto mask = stride * 2 - 1;
        const auto half_mask = mask >> 1;

        for (std::uint32_t j = block.thread_rank(); j < values_count / 2; j += block.size())
        {
            // skip odd blocks (assuming blocks of size stride)
            const auto i = ((j & ~half_mask) * 2) + (j & half_mask);
            const auto target = i ^ mask;
            assert(target > i);

            const bool swap = ORDER == order_t::ascending ? values.dist(i) > values.dist(target)
                                                          : values.dist(i) < values.dist(target);

            if (swap)
            {
                values.swap(i, target);
            }
        }
    }
    else
    {
        for (std::uint32_t i = block.thread_rank(); i < values_count; i += block.size())
        {
            auto local_dist = values.dist(i);
            auto local_label = values.label(i);

            warp_reversed_bitonic_stage<float, std::int32_t, ORDER>(local_dist, local_label,
                                                                    stride);

            // store the sorted distances to shared memory
            values.dist(i) = local_dist;
            values.label(i) = local_label;
        }
    }
    __syncthreads();
}

/** Sort a sequence using Bitonic sort.
 *
 * @tparam USE_WARP_SORT if true, use warp shuffles for small strides (<= 16).
 * @tparam STRIDE first stride of the Bitonic sort algorithm
 * @tparam ORDER ascending or descending order.
 * @tparam Layout memory layout (structure-of-arrays or arrays-of-structures).
 * @param values array of values to sort
 * @param values_count number of values in the `values` array
 */
template <bool USE_WARP_SORT, std::uint32_t STRIDE, order_t ORDER = order_t::ascending,
          typename Layout = soa_layout>
__device__ void block_sort(Layout values, std::uint32_t values_count)
{
    for (std::uint32_t stride = STRIDE; stride < values_count; stride *= 2)
    {
        block_merge<USE_WARP_SORT, ORDER, Layout>(values, stride, values_count);
        block_sort_bitonic<USE_WARP_SORT, ORDER, Layout>(values, stride / 2, values_count);
    }
}

#endif // BITONIC_SORT_CUH_
