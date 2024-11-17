#ifndef BITONIC_SORT_WARP_CUH_
#define BITONIC_SORT_WARP_CUH_

#include <cassert>
#include <cstddef>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

enum order_t
{
    ascending,
    descending
};

/** Single bitonic merge step within each warp
 *
 * @param[in,out] key Value used for sorting
 * @param[in,out] value Value assigned to @p key
 * @param stride stride of the bitonic step (i.e., subsequences of size `stride` are sorted)
 */
template <typename Key, typename Value, order_t ORDER = order_t::ascending>
__device__ __forceinline__ void warp_reversed_bitonic_stage(Key& key, Value& value,
                                                            std::size_t stride)
{
    namespace cg = cooperative_groups;

    constexpr std::size_t WARP_SIZE = 32;

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);

    // merge consecutive sorted sequences of size `stride`
    const auto mask = stride * 2 - 1;
    const auto other_key = warp.shfl_xor(key, mask);
    const auto other_value = warp.shfl_xor(value, mask);
    const bool is_rhs = (warp.thread_rank() ^ mask) < warp.thread_rank();

    const auto pred = is_rhs ^ (ORDER == order_t::ascending) ? key > other_key : key < other_key;

    key = pred ? other_key : key;
    value = pred ? other_value : value;
}

/** Single step of a bitonic sort within each warp
 *
 * @param[in,out] key Value used for sorting
 * @param[in,out] value Value assigned to @p key
 * @param stride stride of the bitonic step (i.e., subsequences of size `stride` are bitonic)
 */
template <typename Key, typename Value, order_t ORDER = order_t::ascending>
__device__ __forceinline__ void warp_sort_bitonic_step(Key& key, Value& value, std::size_t stride)
{
    namespace cg = cooperative_groups;

    constexpr std::size_t WARP_SIZE = 32;

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);

    const auto other_key = warp.shfl_xor(key, stride);
    const auto other_value = warp.shfl_xor(value, stride);
    const bool is_rhs = (warp.thread_rank() & stride);

    const auto pred = is_rhs ^ (ORDER == order_t::ascending) ? key > other_key : key < other_key;

    key = pred ? other_key : key;
    value = pred ? other_value : value;
}

/** Sort bitonic subsequences within each warp
 *
 * @param[in,out] key Value used for sorting
 * @param[in,out] value Value assigned to @p key
 * @param stride stride of the bitonic step (i.e., subsequences of size `stride` are bitonic)
 */
template <typename Key, typename Value, order_t ORDER = order_t::ascending>
__device__ __forceinline__ void warp_sort_bitonic(Key& key, Value& value, std::size_t stride)
{
    namespace cg = cooperative_groups;

    constexpr std::size_t WARP_SIZE = 32;

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);

// sort consecutive bitonic sequences of size `stride`
#pragma unroll
    for (; stride > 0; stride /= 2)
    {
        const auto other_key = warp.shfl_xor(key, stride);
        const auto other_value = warp.shfl_xor(value, stride);
        const bool target = warp.thread_rank() & stride;

        const auto pred =
            target ^ (ORDER == order_t::ascending) ? key > other_key : key < other_key;

        key = pred ? other_key : key;
        value = pred ? other_value : value;
    }
}

/** Sort values in a warp. Each thread in a warp has a single pair (distance, index).
 *
 * @param[in,out] key Value used for sorting
 * @param[in,out] value Value assigned to @p key
 */
template <typename Key, typename Value, order_t order = order_t::ascending>
__device__ __forceinline__ void warp_sort(Key& key, Value& value)
{
    constexpr std::size_t WARP_SIZE = 32;

#pragma unroll
    for (std::size_t stride = 1; stride < WARP_SIZE; stride *= 2)
    {
        warp_reversed_bitonic_stage<Key, Value, order>(key, value, stride);
        warp_sort_bitonic<Key, Value, order>(key, value, stride / 2);
    }
}

#endif // BITONIC_SORT_WARP_CUH_
