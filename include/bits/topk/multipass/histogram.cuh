#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cub/block/block_scan.cuh>

#include "bits/block_array_utils.cuh"

/** Compute a histogram in shared memory.
 *
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam ITEMS_PER_THREAD number of registers in each thread.
 * @tparam BUCKET_BITS log2 of the histogram size.
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, std::size_t BUCKET_BITS>
struct histogram
{
    using bucket_t = std::conditional_t<BUCKET_BITS <= 8, std::uint8_t, std::uint16_t>;

    // total number of buckets
    static constexpr std::size_t BUCKET_COUNT = 1 << BUCKET_BITS;
    // total size of a tile
    static constexpr std::size_t TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
    // number of buckets per thread
    static constexpr std::size_t BUCKETS_PER_THREAD =
        (BUCKET_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    using scan_t = cub::BlockScan<std::int32_t, BLOCK_SIZE>;

    /** Shared memory used by the histogram.
     */
    union tmp_storage_t
    {
        // shared memory for counters
        std::int32_t counter[BUCKET_COUNT];
        // shared memory for the prefix sum
        typename scan_t::TempStorage scan;

        // shared memory for bucket()
        struct
        {
            std::int32_t lower_bound;
            std::int32_t begin;
            std::int32_t end;
        };
    };

    tmp_storage_t& tmp_storage;

    /** Create a histogram
     *
     * @param storage shared memory storage reference
     */
    __device__ __forceinline__ explicit histogram(tmp_storage_t& storage) : tmp_storage(storage) {}

    /** Reset histogram counters
     */
    __device__ __forceinline__ void reset()
    {
        for (std::size_t i = threadIdx.x; i < BUCKET_COUNT; i += BLOCK_SIZE)
        {
            tmp_storage.counter[i] = 0;
        }
        __syncthreads();
    }

    /** Increment @p bucket by 1
     *
     * @param bucket bucket to increment (in the [0, BUCKET_COUNT) range)
     */
    __device__ __forceinline__ void inc(bucket_t bucket)
    {
        atomicAdd(&tmp_storage.counter[bucket], 1);
    }

    /** Increment @p buckets by 1
     *
     * @param buckets buckets to increment (in the [0, BUCKET_COUNT) range)
     */
    __device__ __forceinline__ void inc(bucket_t (&buckets)[ITEMS_PER_THREAD])
    {
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            atomicAdd(&tmp_storage.counter[buckets[i]], 1);
        }
    }

    /** Read values from a given segment and increment corresponding counters
     *
     * @tparam Splitter class used to split keys to buckets
     *
     * @param keys_ptr Base pointer to memory with values
     * @param segment_begin Begin index of range in @p keys_ptr
     * @param segment_end End index of range in @p keys_ptr
     * @param splitter class used to split keys to buckets
     */
    template <class Splitter>
    __device__ __forceinline__ void process(const float* keys_ptr, std::size_t segment_begin,
                                            std::size_t segment_end, Splitter& splitter)
    {
        float keys[ITEMS_PER_THREAD];
        for (; segment_begin + TILE_SIZE < segment_end; segment_begin += TILE_SIZE)
        {
            // load values from global memory
            load_striped<BLOCK_SIZE>(keys, keys_ptr + segment_begin);

            // find bucket for each value
            bucket_t buckets[ITEMS_PER_THREAD];
            splitter.bucket(keys, buckets);

            // increment the bucket counts in shared memory
            inc(buckets);
        }

        if (segment_begin < segment_end)
        {
            // load values from global memory
            load_striped<BLOCK_SIZE>(keys, keys_ptr + segment_begin, segment_end - segment_begin);

            // find bucket for each value
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                const auto idx = segment_begin + i * BLOCK_SIZE + threadIdx.x;
                if (idx < segment_end)
                {
                    inc(splitter.bucket(keys[i]));
                }
            }
        }

        __syncthreads();
    }

    /** Compute a prefix sum of values in the histogram
     *
     * @param[out] offsets Block-wide register array with the prefix sum
     */
    __device__ __forceinline__ void extract_offsets(std::int32_t (&offsets)[BUCKETS_PER_THREAD])
    {
        // load counts to registers
#pragma unroll
        for (std::size_t i = 0; i < BUCKETS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x * BUCKETS_PER_THREAD + i;
            offsets[i] = idx < BUCKET_COUNT ? tmp_storage.counter[idx] : 0;
        }

        __syncthreads();

        // perform prefix sum
        scan_t scan{tmp_storage.scan};
        scan.ExclusiveSum(offsets, offsets);

        __syncthreads();
    }

    /** Find a bucket which contains @p idx
     *
     * @param[in,out] idx Searched index relative to @p begin (i.e., idx = 0 points to @p begin ).
     * Offset of the bucket of @p idx will be subtracted from @p idx
     * @param[in,out] begin Begin index of the whole range. This function will set this value to the
     * beginning of the bucket which contains @p idx
     * @param[in,out] end End index of the whole range. This function will set this value to the
     * end of the bucket which contains @p idx
     * @return index of the bucket which contains @p idx
     */
    __device__ __forceinline__ std::int32_t bucket(std::int32_t& idx, std::int32_t& begin,
                                                   std::int32_t& end)
    {
        std::int32_t offsets[BUCKETS_PER_THREAD];
        extract_offsets(offsets);

        // find bucket which contains `idx`
        const auto lb_idx =
            lower_bound<BLOCK_SIZE>(idx, offsets, BUCKET_COUNT, &tmp_storage.lower_bound);

// update offsets to global offsets
#pragma unroll
        for (std::size_t i = 0; i < BUCKETS_PER_THREAD; ++i)
        {
            offsets[i] += begin;
        }

        // find segment boundaries of the bucket that contains k
        tmp_storage.end = end;

        __syncthreads();

#pragma unroll
        for (std::size_t i = 0; i < BUCKETS_PER_THREAD; ++i)
        {
            const auto bin_idx = threadIdx.x * BUCKETS_PER_THREAD + i;
            if (bin_idx == lb_idx)
            {
                tmp_storage.begin = offsets[i];
            }
            else if (bin_idx == lb_idx + 1)
            {
                tmp_storage.end = offsets[i];
            }
        }

        __syncthreads();

        const auto old_begin = begin;
        begin = tmp_storage.begin;
        end = tmp_storage.end;
        idx = idx - (begin - old_begin);

        __syncthreads();

        return lb_idx;
    }
};

#endif // HISTOGRAM_CUH_
