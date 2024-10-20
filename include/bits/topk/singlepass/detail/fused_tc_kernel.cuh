#ifndef DETAIL_FUSED_TC_KERNEL_CUH_
#define DETAIL_FUSED_TC_KERNEL_CUH_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "bits/array_view.hpp"
#include "bits/knn.hpp"
#include "bits/topk/singlepass/fused_tc_kernel_runner.hpp"
#include "bits/topk/singlepass/fused_tc_policy.hpp"

#include "bits/topk/bitonic_sort_regs.cuh"

namespace
{

/** Merge `ARRAY_COUNT` buffers in shared memory with top k results.
 *
 * @tparam K size of each array/buffer.
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam ITEMS_PER_THREAD number of items stored by each thread.
 * @tparam ARRAY_COUNT number of arrays/buffers in each thread block.
 * @param[in,out] dist block-wide register array which contains the top k distances from
 * `ARRAY_COUNT` arrays.
 * @param[in,out] label block-wide register array which contains the top k labels from `ARRAY_COUNT`
 * arrays.
 * @param[in] shm_dist shared memory with the buffers for distances.
 * @param[in] shm_label shared memory with the buffers for labels.
 * @param buffer_size buffer size for each of the `ARRAY_COUNT` buffers (in shared memory).
 */
template <std::int32_t K, std::int32_t BLOCK_SIZE, std::int32_t ITEMS_PER_THREAD,
          std::int32_t ARRAY_COUNT>
__device__ void merge_tc_buffers(float (&dist)[ITEMS_PER_THREAD],
                                 std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                                 std::int32_t* shm_label, std::int32_t* buffer_size)
{
    static_assert(ITEMS_PER_THREAD <= K);
    static_assert(ITEMS_PER_THREAD * BLOCK_SIZE >= K);

    // total number of values
    constexpr std::int32_t VALUE_COUNT = ARRAY_COUNT * K;

    float buffer_dist[ITEMS_PER_THREAD];
    std::int32_t buffer_label[ITEMS_PER_THREAD];

    const auto block = cooperative_groups::this_thread_block();

#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        buffer_dist[i] = std::numeric_limits<float>::infinity();
    }

#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // linear access with bank conflicts (assuming row-major layout)
        auto buffer_idx = block.thread_rank() * ITEMS_PER_THREAD + i;
        // conflict-free access pattern (assuming interleaved layout)
        // auto buffer_idx = block.thread_rank() + i * BLOCK_SIZE;

        if (buffer_idx < VALUE_COUNT)
        {
            buffer_dist[i] = shm_dist[buffer_idx];
            buffer_label[i] = shm_label[buffer_idx];
        }
    }

    __syncthreads();

    // sort all buffers
    block_sort_partial<1, K, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, order_t::descending>(
        buffer_dist, buffer_label, shm_dist, shm_label);

// merge buffers with the top k lists
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        if (dist[i] > buffer_dist[i])
        {
            dist[i] = buffer_dist[i];
            label[i] = buffer_label[i];
        }
    }

    // sort the lower half of the values
    block_sort_bitonic<K / 2, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD>(dist, label, shm_dist,
                                                                         shm_label);

    // reset buffer size
    for (std::int32_t i = block.thread_rank(); i < ARRAY_COUNT; i += BLOCK_SIZE)
    {
        buffer_size[i] = std::max<std::int32_t>(buffer_size[i] - K, 0);
    }

    __syncthreads();
}

template <std::int32_t BLOCK_SIZE, typename Policy, std::int32_t LAYOUT>
__global__ void prepare_points(array_view<float, 2> points,
                               array_view<typename Policy::input_t, 2> out_points,
                               array_view<float, 1> out_norms)
{
    const auto idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const auto dim = points.size(1);
    const auto aligned_dim =
        (dim + Policy::DIM_TILE_SIZE - 1) / Policy::DIM_TILE_SIZE * Policy::DIM_TILE_SIZE;
    const auto dim_tiles = aligned_dim / Policy::DIM_TILE_SIZE;

    // fill the padding with neutral elements
    if (idx >= points.size(0))
    {
        if (idx < out_norms.size())
        {
            out_norms(idx) = std::numeric_limits<float>::infinity();
        }

        for (std::int32_t i = 0; i < aligned_dim; ++i)
        {
            const auto tile_idx = idx / LAYOUT * dim_tiles + i / Policy::DIM_TILE_SIZE;
            const auto tile_offset =
                idx % LAYOUT * Policy::DIM_TILE_SIZE + i % Policy::DIM_TILE_SIZE;

            if (tile_idx < out_points.size(0))
            {
                out_points(tile_idx, tile_offset) = Policy::zero_input();
            }
        }

        return;
    }

    // cast and rearrange the points and compute the norms
    double sum = 0.0;
    for (std::int32_t i = 0; i < aligned_dim; ++i)
    {
        const auto tile_idx = idx / LAYOUT * dim_tiles + i / Policy::DIM_TILE_SIZE;
        const auto tile_offset = idx % LAYOUT * Policy::DIM_TILE_SIZE + i % Policy::DIM_TILE_SIZE;

        const auto value = i < dim ? points(idx, i) : 0.0f;
        sum = fma(value, value, sum);

        out_points(tile_idx, tile_offset) = Policy::from_float(value);
    }

    out_norms(idx) = sum;
}

/** Modified fused regs kernel with tensor cores.
 * - see: "Detailed Analysis and Optimization of CUDA K-means, Krulis et al."
 * -
 * https://github.com/krulis-martin/cuda-kmeans/blob/master/experimental/k-means/k-means/kernels/kernels.cu
 *
 * @tparam DIM_BLOCK number of unrolled iterations of the dimension loop
 * @tparam K size of the output
 * @tparam BLOCK_QUERY_DIM number of threads in each thread block along the query dimension.
 * @tparam BLOCK_POINT_DIM number of threads in each thread block along the point dimension.
 * @param[in] queries dimension * number of queries matrix of query vectors.
 * @param[in] points dimension * number of points matrix of database vectors.
 * @param[out] out_dist top k distance for each query.
 * @param[out] out_label top k label for each query.
 * @param[in] dim dimension of all vectors in @p queries and @p points
 * @param[in] point_count number of database vectors.
 * @param[in] query_count number of query vectors.
 */
template <std::int32_t K, std::int32_t QUERY_BATCH, std::int32_t POINT_BATCH,
          std::int32_t BLOCK_SIZE, typename Policy>
__global__ void
fused_tc_kernel(array_view<typename Policy::input_t, 2> points, array_view<float, 1> point_norms,
                array_view<typename Policy::input_t, 2> queries, array_view<float, 1> query_norms,
                array_view<float, 2> out_dist, array_view<std::int32_t, 2> out_label,
                std::size_t dim, std::size_t point_count, std::size_t query_count)
{
#ifdef __CUDA_ARCH__
    // not supported
    constexpr std::int32_t CUDA_ARCH{__CUDA_ARCH__};
#else
    constexpr std::int32_t CUDA_ARCH{0};
#endif
    if constexpr (CUDA_ARCH < 800 && (std::is_same<Policy, fused_tc_bfloat16_policy>::value ||
                                      std::is_same<Policy, fused_tc_double_policy>::value))
    {
        // not supported
        asm volatile("trap;");
    }
    else
    {
        assert(blockDim.x == QUERY_BATCH);
        assert(blockDim.y == POINT_BATCH);

        constexpr std::int32_t QUERY_TILE_SIZE = Policy::QUERY_TILE_SIZE;
        constexpr std::int32_t POINT_TILE_SIZE = Policy::POINT_TILE_SIZE;
        constexpr std::int32_t DIM_TILE_SIZE = Policy::DIM_TILE_SIZE;

        using input_t = typename Policy::input_t;
        using output_t = typename Policy::output_t;

        // so we can comfortably use the 8x16x32 matrix multiply:
        static_assert(QUERY_BATCH % QUERY_TILE_SIZE == 0);
        static_assert(POINT_BATCH % POINT_TILE_SIZE == 0);

        static_assert(BLOCK_SIZE % QUERY_BATCH == 0);
        static_assert(BLOCK_SIZE % POINT_BATCH == 0);

        constexpr std::int32_t QUERY_BATCH_TILES = QUERY_BATCH / QUERY_TILE_SIZE;
        constexpr std::int32_t POINT_BATCH_TILES = POINT_BATCH / POINT_TILE_SIZE;

        constexpr std::int32_t QTILE_SIZE = QUERY_TILE_SIZE * DIM_TILE_SIZE;
        constexpr std::int32_t PTILE_SIZE = POINT_TILE_SIZE * DIM_TILE_SIZE;

        // total size of all buffers stored in each thread block
        constexpr std::int32_t BUFFER_SIZE =
            std::max<std::int32_t>(BLOCK_SIZE / QUERY_BATCH, K) * QUERY_BATCH;

        // number of items from top k lists stored per thread
        constexpr std::int32_t ITEMS_PER_THREAD = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

        constexpr std::int32_t WARP_SIZE = 32;

        // type to store buffer positions
        using position_t = std::int32_t;

        struct buffer_t
        {
            // buffer
            float dist[BUFFER_SIZE];
            std::int32_t label[BUFFER_SIZE];
            // buffer size for each buffer
            std::int32_t size[QUERY_BATCH];
        };

        __shared__ buffer_t buffer;

        // shared memory for the results of the matrix multiplication
        __shared__ output_t shm_results[QUERY_BATCH * POINT_BATCH];

        const auto block = cooperative_groups::this_thread_block();
        const auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);

        const std::int32_t aligned_dim = (dim + DIM_TILE_SIZE - 1) / DIM_TILE_SIZE * DIM_TILE_SIZE;
        const std::int32_t dim_tiles = aligned_dim / DIM_TILE_SIZE;

        const std::int32_t qtensor = warp.meta_group_rank() % QUERY_BATCH_TILES * QUERY_TILE_SIZE;
        const std::int32_t ptensor = warp.meta_group_rank() / QUERY_BATCH_TILES * POINT_TILE_SIZE;

        // initialize buffer size
        for (std::int32_t i = block.thread_rank(); i < QUERY_BATCH; i += BLOCK_SIZE)
        {
            buffer.size[i] = 0;
        }

        // reset buffer
        for (std::int32_t i = block.thread_rank(); i < BUFFER_SIZE; i += BLOCK_SIZE)
        {
            buffer.dist[i] = std::numeric_limits<float>::infinity();
        }

        __syncthreads();

        // initialize radius
        float radius[QUERY_BATCH];
#pragma unroll
        for (std::int32_t rq = 0; rq < QUERY_BATCH; ++rq)
        {
            radius[rq] = std::numeric_limits<float>::infinity();
        }

        // portion of the top k lists stored in this thread
        float topk_dist[ITEMS_PER_THREAD];
        std::int32_t topk_label[ITEMS_PER_THREAD];

// reset the top k lists
#pragma unroll
        for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            topk_dist[i] = std::numeric_limits<float>::infinity();
        }

        // distance computation
        for (std::int32_t points_offset = 0; points_offset < static_cast<std::int32_t>(point_count);
             points_offset += POINT_BATCH)
        {
            namespace wmma = nvcuda::wmma;

            wmma::fragment<wmma::accumulator, QUERY_TILE_SIZE, POINT_TILE_SIZE, DIM_TILE_SIZE,
                           output_t>
                acc_frag;
            wmma::fragment<wmma::matrix_a, QUERY_TILE_SIZE, POINT_TILE_SIZE, DIM_TILE_SIZE, input_t,
                           wmma::row_major>
                a_frag;
            wmma::fragment<wmma::matrix_b, QUERY_TILE_SIZE, POINT_TILE_SIZE, DIM_TILE_SIZE, input_t,
                           wmma::col_major>
                b_frag;

            // ensure all warps are busy
            static_assert(BLOCK_SIZE / 32 == QUERY_BATCH_TILES * POINT_BATCH_TILES);

            // compute a fragment sized QUERY_TILE_SIZE x POINT_TILE_SIZE
            wmma::fill_fragment(acc_frag, Policy::zero_output());

            for (std::int32_t k = 0; k < aligned_dim; k += DIM_TILE_SIZE)
            {
                const auto pidx = points_offset + ptensor;
                const auto qidx = blockIdx.x * QUERY_BATCH + qtensor;

                const auto ptile = pidx / POINT_TILE_SIZE * dim_tiles + k / DIM_TILE_SIZE;
                const auto qtile = qidx / QUERY_TILE_SIZE * dim_tiles + k / DIM_TILE_SIZE;

                wmma::load_matrix_sync(a_frag, queries.data() + qtile * QTILE_SIZE, DIM_TILE_SIZE);
                wmma::load_matrix_sync(b_frag, points.data() + ptile * PTILE_SIZE, DIM_TILE_SIZE);

                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            wmma::store_matrix_sync(shm_results + qtensor + ptensor * QUERY_BATCH, acc_frag,
                                    QUERY_BATCH, wmma::mem_col_major);

            constexpr std::int32_t BATCH_PER_THREAD = QUERY_BATCH * POINT_BATCH / BLOCK_SIZE;

            // allocate buffer spots for all computed distances
            float dist[BATCH_PER_THREAD];
            position_t buffer_pos[BATCH_PER_THREAD];

#pragma unroll
            for (std::int32_t i = 0; i < BATCH_PER_THREAD; ++i)
            {
                const auto idx = warp.thread_rank() * BATCH_PER_THREAD + i;
                const auto query_idx = idx % QUERY_TILE_SIZE + qtensor;
                const auto point_idx = idx / QUERY_TILE_SIZE + ptensor;

                const float result =
                    Policy::to_float(shm_results[query_idx + point_idx * QUERY_BATCH]);

                // const float distance = point_norms(points_offset + point_idx) - 2.f * result;
                dist[i] = fmaf(-2.f, result, point_norms(points_offset + point_idx));
                buffer_pos[i] =
                    dist[i] < radius[query_idx] ? atomicAdd(&buffer.size[query_idx], 1) : -1;
            }

            // merge buffers if necessary
            for (;;)
            {
                bool overflown = false;

#pragma unroll
                for (std::int32_t i = 0; i < BATCH_PER_THREAD; ++i)
                {
                    const auto idx = warp.thread_rank() * BATCH_PER_THREAD + i;
                    const auto query_idx = idx % QUERY_TILE_SIZE + qtensor;
                    const auto point_idx = idx / QUERY_TILE_SIZE + ptensor;

                    const auto buffer_idx = buffer_pos[i] + query_idx * K;

                    const auto point_offset = points_offset + point_idx;

                    if (0 <= buffer_pos[i] && buffer_pos[i] < K)
                    {
                        buffer.dist[buffer_idx] = dist[i];
                        buffer.label[buffer_idx] = point_offset;
                    }

                    // check for overflow
                    overflown |= buffer_pos[i] >= static_cast<position_t>(K);

                    // decrement buffer position for the next iteration.
                    buffer_pos[i] -= static_cast<position_t>(K);
                }

                // if no thread filled a buffer, we can continue without merging
                if (!__syncthreads_or(overflown))
                {
                    break;
                }

                // merge the buffers
                merge_tc_buffers<K, BLOCK_SIZE, ITEMS_PER_THREAD, QUERY_BATCH>(
                    topk_dist, topk_label, buffer.dist, buffer.label, buffer.size);

#pragma unroll
                // store radii of all lists to shared memory
                for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
                {
                    const auto idx = block.thread_rank() * ITEMS_PER_THREAD + i;
                    const auto array_idx = idx / K;
                    const auto element_idx = idx % K;
                    if (element_idx + 1 >= K)
                    {
                        buffer.dist[array_idx] = topk_dist[i];
                    }
                }

                __syncthreads();

#pragma unroll
                // update radii
                for (std::int32_t i = 0; i < QUERY_BATCH; ++i)
                {
                    radius[i] = buffer.dist[i];
                }

                __syncthreads();

#pragma unroll
                // reset buffers
                for (std::int32_t i = block.thread_rank(); i < BUFFER_SIZE; i += BLOCK_SIZE)
                {
                    buffer.dist[i] = std::numeric_limits<float>::infinity();
                }

                __syncthreads();
            }
        }

        // one final merge of all buffers
        merge_tc_buffers<K, BLOCK_SIZE, ITEMS_PER_THREAD, QUERY_BATCH>(
            topk_dist, topk_label, buffer.dist, buffer.label, buffer.size);

#pragma unroll
        // store the results to global memory
        for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = block.thread_rank() * ITEMS_PER_THREAD + i;
            const auto array_idx = idx / K;
            const auto element_idx = idx % K;
            const auto query_idx = array_idx + blockIdx.x * QUERY_BATCH;

            if (query_idx < query_count)
            {
                out_dist(query_idx, element_idx) = topk_dist[i] + query_norms(query_idx);
                out_label(query_idx, element_idx) = topk_label[i];
            }
        }
    }
}

} // namespace

/** Run the fused tensor core kernel.
 */
template <typename Policy>
template <std::int32_t K, std::int32_t BLOCK_SIZE>
void fused_tc_kernel_runner<Policy>::operator()()
{

    const auto dim = points.size(1);
    const auto point_count = points.size(0);
    const auto query_count = queries.size(0);

    constexpr auto PTILE_SIZE = Policy::POINT_TILE_SIZE * Policy::DIM_TILE_SIZE;
    constexpr auto QTILE_SIZE = Policy::QUERY_TILE_SIZE * Policy::DIM_TILE_SIZE;

    const auto aligned_point_count = (point_count + PTILE_SIZE - 1) / PTILE_SIZE * PTILE_SIZE;
    const auto aligned_query_count = (query_count + QTILE_SIZE - 1) / QTILE_SIZE * QTILE_SIZE;

    constexpr std::int32_t QUERY_BATCH = Policy::QUERY_TILE_SIZE;
    constexpr std::int32_t POINT_BATCH =
        BLOCK_SIZE / 32 * Policy::POINT_TILE_SIZE * Policy::QUERY_TILE_SIZE / QUERY_BATCH;

    constexpr dim3 block(QUERY_BATCH, BLOCK_SIZE / QUERY_BATCH);
    const dim3 grid((query_count + QUERY_BATCH - 1) / QUERY_BATCH, 1);

    // prepare the input
    prepare_points<256, Policy, Policy::POINT_TILE_SIZE>
        <<<(aligned_point_count + 255) / 256, 256>>>(points, in_points, in_point_norms);
    prepare_points<256, Policy, Policy::QUERY_TILE_SIZE>
        <<<(aligned_query_count + 255) / 256, 256>>>(queries, in_queries, in_query_norms);

    // call the kernel
    fused_tc_kernel<K, QUERY_BATCH, POINT_BATCH, BLOCK_SIZE, Policy>
        <<<grid, block>>>(in_points, in_point_norms, in_queries, in_query_norms, out_dist,
                          out_label, dim, point_count, query_count);
}

#endif // DETAIL_FUSED_TC_KERNEL_CUH_
