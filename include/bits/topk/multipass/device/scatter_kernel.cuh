#ifndef SCATTER_KERNEL_CUH_
#define SCATTER_KERNEL_CUH_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/block/block_scan.cuh>

#include "bits/memory.cuh"
#include "bits/topk/multipass/radix_splitter.cuh"
#include "bits/topk/multipass/splitter.cuh"

/** A modified radix rank and downsweep from the CUB library:
 * https://github.com/NVIDIA/cub/blob/main/cub/block/block_radix_rank.cuh
 *
 * It has a privitized histogram for each warp. Each thread computes the rank within its warp and
 * updates its warp-private histogram in shared memory. An exclusive previx sum of the whole table
 * of privatized histograms is computed which gives the threads offset within the thread block.
 * This offset is added to the global offset.
 *
 * Computed ranks are used to sort the values within the thread block to minimize the number of
 * global memory transactions when the values are scattered to the global memory.
 *
 * @tparam BLOCK_SIZE_ number of threads in each thread block.
 * @tparam ITEMS_PER_THREAD number of registers per thread.
 * @tparam BIN_BITS log2 of the histogram size.
 * @tparam TILES_PER_BLOCK number of tiles of size `BLOCK_SIZE * ITEMS_PER_THREAD` processed by
 * each thread block.
 * @tparam Splitter splitter used to split distances (floats) to histogram buckets.
 */
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
template <std::size_t BLOCK_SIZE_, std::size_t ITEMS_PER_THREAD, std::size_t BIN_BITS,
          std::size_t TILES_PER_BLOCK, class Splitter>
struct scatter_kernel
{
    static_assert(BIN_BITS <= 16);

    // number of threads in each thread block
    inline static constexpr std::size_t BLOCK_SIZE = BLOCK_SIZE_;
    // number of items per thread block
    inline static constexpr std::size_t ITEMS_PER_BLOCK =
        BLOCK_SIZE * ITEMS_PER_THREAD * TILES_PER_BLOCK;
    // log of the warp size
    inline static constexpr std::size_t LOG_WARP_SIZE = 5;
    // number of histogram bins per thread
    inline static constexpr std::size_t NUM_WARPS =
        BLOCK_SIZE >> LOG_WARP_SIZE; // block size / warp size
    inline static constexpr std::size_t PADDED_NUM_WARPS = NUM_WARPS + 3;
    // number of bins in a histogram
    inline static constexpr std::size_t HIST_SIZE = 1 << BIN_BITS;
    // bitmask representing the whole warp
    inline static constexpr std::uint32_t WHOLE_WARP = 0xFFFFFFFF;
    // number of privatized histogram table values per thread (when stored in registers of a thread
    // block)
    inline static constexpr std::size_t COUNTERS_PER_THREAD =
        (HIST_SIZE * NUM_WARPS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    inline static constexpr std::size_t PADDED_COUNTERS_PER_THREAD =
        (HIST_SIZE * PADDED_NUM_WARPS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // number of histogram bins per thread (when stored in registers of a thread block)
    inline static constexpr std::size_t BINS_PER_THREAD = (HIST_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    using scan_t = cub::BlockScan<std::int32_t, BLOCK_SIZE>;
    using splitter_t = Splitter;
    using bucket_t = typename splitter_t::bucket_t;
    using histogram_t = histogram<BLOCK_SIZE, ITEMS_PER_THREAD, BIN_BITS>;

    /** Shared memory used by this kernel.
     */
    struct tmp_storage_t
    {
        union
        {
            // arrays for permuting values in a tile
            float permute_dist[BLOCK_SIZE * ITEMS_PER_THREAD];
            std::int32_t permute_label[BLOCK_SIZE * ITEMS_PER_THREAD];
            // privatized histograms (one per warp) for ranking (the idea is from the CUB library)
            volatile std::int32_t priv_hist[HIST_SIZE][PADDED_NUM_WARPS];
            std::int32_t priv_hist_block[BLOCK_SIZE][PADDED_COUNTERS_PER_THREAD];
            // memory for the prefix sum
            typename scan_t::TempStorage scan;
            // an exclusive prefix sum of a histogram
            std::int32_t digit_scan[HIST_SIZE];
        };

        // offset of each bin
        std::int32_t offset[HIST_SIZE];
        // splitter state in shraed memory
        typename splitter_t::tmp_storage_t splitter;
    };

    tmp_storage_t* tmp_storage;

    // input list of distances
    float* global_dist_in;
    // input list of labels
    std::int32_t* global_label_in;
    // output list of distances
    float* global_dist_out;
    // output list of labels
    std::int32_t* global_label_out;
    // number of pairs in the input
    std::int32_t input_size;
    // prefix sum of the histogram table
    std::int32_t* global_hist;
    // state of the splitter
    typename splitter_t::state_t splitter_state;

    __device__ __forceinline__ void set_tmp_storage(tmp_storage_t& shm) { tmp_storage = &shm; }

    /** Scatter all distance/label paits in the @p dist @p label tile to the correct buckets in
     * global memory.
     *
     * @tparam CHECK_BOUNDS if true, only the first @p tile_size values from the tile are
     * considered.
     * @param dist distances in the tile in the strided layout (i.e., index of `dist[i]` in
     * thread `tid` is `tid + i * BLOCK_SIZE`).
     * @param label labels in the tile in the strided layout -- same as @p dist
     * @param offset global histogram bin offsets loaded from global memory.
     * @param tile_size number of valid values in the @p dist @p label tile.
     * @param splitter splitter used to split distances to histogram buckets.
     */
    template <bool CHECK_BOUNDS>
    __device__ __forceinline__ void
    process(float (&dist)[ITEMS_PER_THREAD], std::int32_t (&label)[ITEMS_PER_THREAD],
            std::int32_t (&offset)[BINS_PER_THREAD], std::int32_t tile_size, splitter_t& splitter)
    {
        // rank of each distance within this tile
        std::int32_t rank[ITEMS_PER_THREAD];

// reset privatized histograms
#pragma unroll
        for (std::size_t i = 0; i < COUNTERS_PER_THREAD; ++i)
        {
            tmp_storage->priv_hist_block[threadIdx.x][i] = 0;
        }

        __syncthreads();

        bucket_t bucket[ITEMS_PER_THREAD];
        splitter.bucket(dist, bucket);

        if (CHECK_BOUNDS)
        {
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                const auto idx = i * BLOCK_SIZE + threadIdx.x;
                if (idx >= tile_size)
                {
                    bucket[i] = HIST_SIZE - 1;
                }
            }
        }

// compute the rank and offset
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = i * BLOCK_SIZE + threadIdx.x;
            // find the number of threads with the same digit
            std::uint32_t same_digit = __ballot_sync(WHOLE_WARP, idx < tile_size);
#pragma unroll
            for (std::size_t bit = 0; bit < BIN_BITS; ++bit)
            {
                std::uint32_t bit_value = (bucket[i] >> bit) & 1;
                std::uint32_t mask = __ballot_sync(WHOLE_WARP, bit_value);
                same_digit &= bit_value ? mask : ~mask;
            }

            // find the previous number of digits within this warp
            std::int32_t prev_digit_count =
                tmp_storage->priv_hist[bucket[i]][threadIdx.x >> LOG_WARP_SIZE];

            // count lower threads within the warp with the same digit
            std::uint32_t lane_id = threadIdx.x & ((1 << LOG_WARP_SIZE) - 1);
            std::uint32_t lanemask_lt = ~(WHOLE_WARP << lane_id);
            rank[i] = prev_digit_count + __popc(same_digit & lanemask_lt);

            __syncwarp();

            // add the count to the histogram of this warp
            if (rank[i] <= prev_digit_count && idx < tile_size)
            {
                tmp_storage->priv_hist[bucket[i]][threadIdx.x >> LOG_WARP_SIZE] =
                    prev_digit_count + __popc(same_digit);
            }

            __syncwarp();
        }

        __syncthreads();

        std::int32_t local_hist[COUNTERS_PER_THREAD];
#pragma unroll
        for (std::size_t i = 0; i < COUNTERS_PER_THREAD; ++i)
        {
            local_hist[i] = tmp_storage->priv_hist_block[threadIdx.x][i];
        }

        __syncthreads();

        scan_t{tmp_storage->scan}.ExclusiveSum(local_hist, local_hist);

        __syncthreads();

#pragma unroll
        for (std::size_t i = 0; i < COUNTERS_PER_THREAD; ++i)
        {
            tmp_storage->priv_hist_block[threadIdx.x][i] = local_hist[i];
        }

        __syncthreads();

// finish the rank computation
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            rank[i] += tmp_storage->priv_hist[bucket[i]][threadIdx.x >> LOG_WARP_SIZE];
        }

        // find the prefix sum of digit counters
        std::int32_t digit_scan[BINS_PER_THREAD];
        std::int32_t digit_scan_inclusive[ITEMS_PER_THREAD];

#pragma unroll
        for (std::size_t i = 0; i < BINS_PER_THREAD; ++i)
        {
            const auto bin_idx = threadIdx.x * BINS_PER_THREAD + i;
            if (bin_idx < HIST_SIZE)
            {
                digit_scan[i] = tmp_storage->priv_hist[bin_idx][0];
                digit_scan_inclusive[i] =
                    bin_idx + 1 < HIST_SIZE ? tmp_storage->priv_hist[bin_idx + 1][0] : tile_size;
            }
        }

        __syncthreads();

// update global offset
#pragma unroll
        for (std::size_t i = 0; i < BINS_PER_THREAD; ++i)
        {
            const auto bin_idx = threadIdx.x * BINS_PER_THREAD + i;
            if (bin_idx < HIST_SIZE)
            {
                offset[i] -= digit_scan[i];
                tmp_storage->offset[bin_idx] = offset[i];
                offset[i] += digit_scan_inclusive[i];
            }
        }

// permute distances in shared memory
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (!CHECK_BOUNDS || idx < tile_size)
            {
                tmp_storage->permute_dist[rank[i]] = dist[i];
            }
        }

        __syncthreads();

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (!CHECK_BOUNDS || idx < tile_size)
            {
                dist[i] = tmp_storage->permute_dist[idx];
            }
        }

        __syncthreads();

// premute labels
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (!CHECK_BOUNDS || idx < tile_size)
            {
                tmp_storage->permute_label[rank[i]] = label[i];
            }
        }

        __syncthreads();

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (!CHECK_BOUNDS || idx < tile_size)
            {
                label[i] = tmp_storage->permute_label[idx];
            }
        }

// write the values to the global memory
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = i * BLOCK_SIZE + threadIdx.x;
            if (!CHECK_BOUNDS || idx < tile_size)
            {
                bucket_t digit = splitter.bucket(dist[i]);
                global_dist_out[tmp_storage->offset[digit] + threadIdx.x + i * BLOCK_SIZE] =
                    dist[i];
                global_label_out[tmp_storage->offset[digit] + threadIdx.x + i * BLOCK_SIZE] =
                    label[i];
            }
        }
    }

    /** Load histogram bin offsets from global memory and scatter distance/label pairs in the
     * assigned tiles to the correct histogram bin in global memory.
     *
     * It expects that `global_hist` contains a prefix sum of the histogram table.
     */
    __device__ __forceinline__ void operator()()
    {
        splitter_t splitter{tmp_storage->splitter};
        splitter.load(splitter_state);

        // load bucket offsets from global memory
        std::int32_t offset[BINS_PER_THREAD];

#pragma unroll
        for (std::size_t i = 0; i < BINS_PER_THREAD; ++i)
        {
            const auto digit = threadIdx.x * BINS_PER_THREAD + i;
            if (digit < HIST_SIZE)
            {
                offset[i] = global_hist[blockIdx.x + digit * gridDim.x];
            }
        }

        __syncthreads();

        float dist[ITEMS_PER_THREAD];
        std::int32_t label[ITEMS_PER_THREAD];

        auto it = std::min<std::size_t>(blockIdx.x * ITEMS_PER_BLOCK, input_size);
        const auto end = std::min<std::size_t>(it + ITEMS_PER_BLOCK, input_size);
        for (; it + BLOCK_SIZE * ITEMS_PER_THREAD <= end; it += BLOCK_SIZE * ITEMS_PER_THREAD)
        {
            load_striped<BLOCK_SIZE>(dist, global_dist_in + it);
            load_striped<BLOCK_SIZE>(label, global_label_in + it);
            process<false>(dist, label, offset, BLOCK_SIZE * ITEMS_PER_THREAD, splitter);
        }

        if (it < end)
        {
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                dist[i] = std::numeric_limits<float>::infinity();
            }

            load_striped<BLOCK_SIZE>(dist, global_dist_in + it, end - it);
            load_striped<BLOCK_SIZE>(label, global_label_in + it, end - it);
            process<true>(dist, label, offset, end - it, splitter);
        }
    }
};

#endif // SCATTER_KERNEL_CUH_
