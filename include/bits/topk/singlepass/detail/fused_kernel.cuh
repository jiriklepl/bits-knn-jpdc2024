#ifndef FUSED_KERNEL_CUH_
#define FUSED_KERNEL_CUH_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"
#include "bits/knn.hpp"
#include "bits/topk/singlepass/fused_kernel_runner.hpp"

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
__device__ void merge_buffers(float (&dist)[ITEMS_PER_THREAD],
                              std::int32_t (&label)[ITEMS_PER_THREAD], float* shm_dist,
                              std::int32_t* shm_label, std::int32_t* buffer_size)
{
    static_assert(ITEMS_PER_THREAD <= K);
    static_assert(ITEMS_PER_THREAD * BLOCK_SIZE >= K);

    const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

    // total number of values
    constexpr std::int32_t VALUE_COUNT = ARRAY_COUNT * K;

    float buffer_dist[ITEMS_PER_THREAD];
    std::int32_t buffer_label[ITEMS_PER_THREAD];

#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        buffer_dist[i] = std::numeric_limits<float>::infinity();
    }

#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        // linear access with bank conflicts (assuming row-major layout)
        const auto buffer_idx = thread_idx * ITEMS_PER_THREAD + i;
        // conflict-free access pattern (assuming interleaved layout)
        // auto buffer_idx = thread_idx + i * BLOCK_SIZE;

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
    for (std::int32_t i = thread_idx; i < ARRAY_COUNT; i += BLOCK_SIZE)
    {
        buffer_size[i] = std::max<std::int32_t>(buffer_size[i] - K, 0);
    }

    __syncthreads();
}

/** Modified fused regs kernel
 * - see: "Detailed Analysis and Optimization of CUDA K-means, Krulis et al."
 * -
 * https://github.com/krulis-martin/cuda-kmeans/blob/master/experimental/k-means/k-means/kernels/kernels.cu
 *
 * @tparam REG_POINT_COUNT number of registers used for points
 * @tparam REG_QUERY_COUNT number of registers used for queries
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
template <std::int32_t REG_POINT_COUNT, std::int32_t REG_QUERY_COUNT, std::int32_t DIM_BLOCK,
          std::int32_t K, std::int32_t BLOCK_QUERY_DIM, std::int32_t BLOCK_POINT_DIM>
__global__ void fused_kernel(array_view<float, 2> queries, array_view<float, 2> points,
                             array_view<float, 2> out_dist, array_view<std::int32_t, 2> out_label,
                             std::size_t dim, std::size_t point_count, std::size_t query_count)
{
    assert(blockDim.x == BLOCK_QUERY_DIM);
    assert(blockDim.y == BLOCK_POINT_DIM);

    // number of threads in this thread block
    constexpr std::int32_t BLOCK_SIZE = BLOCK_QUERY_DIM * BLOCK_POINT_DIM;
    // total number of queries and points per block
    constexpr std::int32_t QUERIES_PER_BLOCK = BLOCK_QUERY_DIM * REG_QUERY_COUNT;
    constexpr std::int32_t POINTS_PER_BLOCK = BLOCK_POINT_DIM * REG_POINT_COUNT;
    // total size of all buffers stored in each thread block
    constexpr std::int32_t BUFFER_SIZE = QUERIES_PER_BLOCK * K;
    // number of items from top k lists stored per thread
    constexpr std::int32_t ITEMS_PER_THREAD = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // type to store buffer positions
    using position_t = std::int32_t;

    struct buffer_t
    {
        // buffer
        float dist[BUFFER_SIZE];
        std::int32_t label[BUFFER_SIZE];
        // buffer size for each buffer
        std::int32_t size[QUERIES_PER_BLOCK];
    };

    extern __shared__ std::uint8_t fused_kernel_shm[];

    buffer_t& buffer = reinterpret_cast<buffer_t&>(*fused_kernel_shm);
    // part of the query matrix handled by this thread block
    float* const shm_queries = reinterpret_cast<float*>(fused_kernel_shm + sizeof(buffer_t));
    // window of data from the database point matrix
    float* const shm_points = shm_queries + (dim * QUERIES_PER_BLOCK);

    const std::int32_t thread_idx = threadIdx.y * BLOCK_QUERY_DIM + threadIdx.x;
    const std::int32_t idx = threadIdx.x + blockIdx.x * QUERIES_PER_BLOCK;

    // load chunk of queries assigned to this block (global memory -> shared memory)
    for (std::int32_t d = threadIdx.y; d < static_cast<std::int32_t>(dim); d += BLOCK_POINT_DIM)
    {
        float element[REG_QUERY_COUNT];

#pragma unroll
        for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
        {
            const auto query_idx =
                std::min<std::int32_t>(idx + rq * BLOCK_QUERY_DIM, query_count - 1);
            element[rq] = queries(d, query_idx);
        }

#pragma unroll
        for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
        {
            shm_queries[d * BLOCK_QUERY_DIM * REG_QUERY_COUNT + rq * BLOCK_QUERY_DIM +
                        threadIdx.x] = element[rq];
        }
    }

    // initialize buffer size
    for (std::int32_t i = thread_idx; i < QUERIES_PER_BLOCK; i += BLOCK_SIZE)
    {
        buffer.size[i] = 0;
    }

    // reset buffer
    for (std::int32_t i = thread_idx; i < QUERIES_PER_BLOCK * K; i += BLOCK_SIZE)
    {
        buffer.dist[i] = std::numeric_limits<float>::infinity();
    }

    __syncthreads();

    // initialize radius
    float radius[REG_QUERY_COUNT];
#pragma unroll
    for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
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
         points_offset += POINTS_PER_BLOCK)
    {
        // load the next chunk of database points (global memory -> shared memory)
        const std::int32_t total_points_size = dim * POINTS_PER_BLOCK;
        for (std::int32_t m = thread_idx; m < total_points_size; m += BLOCK_SIZE)
        {
            const auto col = m % POINTS_PER_BLOCK;
            const auto row = m / POINTS_PER_BLOCK;
            const auto point_idx = std::min<std::int32_t>(points_offset + col, point_count - 1);
            shm_points[m] = points(row, point_idx);
        }

        __syncthreads();

        // Compute distance between points
        float sum[REG_QUERY_COUNT][REG_POINT_COUNT];
#pragma unroll
        for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
        {
#pragma unroll
            for (std::int32_t rn = 0; rn < REG_POINT_COUNT; ++rn)
            {
                sum[rq][rn] = 0.0f;
            }
        }

        const float* sd = shm_queries + threadIdx.x;
        const float* sm = shm_points + threadIdx.y;
        for (std::int32_t d = 0; d < static_cast<std::int32_t>(dim); d += DIM_BLOCK)
        {
#pragma unroll
            for (std::int32_t dd = 0; dd < DIM_BLOCK; ++dd)
            {
                // Load data to registers
                float reg_query[REG_QUERY_COUNT];
                float reg_point[REG_POINT_COUNT];

// load components of query vectors (shared memory -> registers)
#pragma unroll
                for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
                {
                    reg_query[rq] = *sd;
                    sd += BLOCK_QUERY_DIM;
                }

// load components of database point vectors (shared memory -> registers)
#pragma unroll
                for (std::int32_t rn = 0; rn < REG_POINT_COUNT; ++rn)
                {
                    reg_point[rn] = *sm;
                    sm += BLOCK_POINT_DIM;
                }

// compute part of the distance matrix for dimension d + dd
#pragma unroll
                for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
                {
#pragma unroll
                    for (std::int32_t rn = 0; rn < REG_POINT_COUNT; ++rn)
                    {
                        const float diff = reg_query[rq] - reg_point[rn];
                        sum[rq][rn] += diff * diff;
                    }
                }
            }
        }

        // allocate buffer spots for all computed distances
        position_t buffer_pos[REG_QUERY_COUNT][REG_POINT_COUNT];

#pragma unroll
        for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
        {
            const auto query_offset = threadIdx.x + rq * BLOCK_QUERY_DIM;

#pragma unroll
            for (std::int32_t rn = 0; rn < REG_POINT_COUNT; ++rn)
            {
                const auto point_idx = points_offset + threadIdx.y + rn * BLOCK_POINT_DIM;

                buffer_pos[rq][rn] = sum[rq][rn] < radius[rq] && point_idx < point_count
                                         ? atomicAdd(&buffer.size[query_offset], 1)
                                         : -1;
            }
        }

        // merge buffers if necessary
        for (;;)
        {
            bool overflown = false;

#pragma unroll
            for (std::int32_t rq = 0; rq < REG_QUERY_COUNT; ++rq)
            {
                const auto query_offset = threadIdx.x + rq * BLOCK_QUERY_DIM;
#pragma unroll
                for (std::int32_t rn = 0; rn < REG_POINT_COUNT; ++rn)
                {
                    // linear access with conflicts in merge_buffers()
                    const auto buffer_idx = query_offset * K + buffer_pos[rq][rn];

                    // interleaved layout without conflicts in merge_buffers()
                    /*
                    constexpr auto THREADS_PER_BUFFER = K / ITEMS_PER_THREAD;
                    static_assert(THREADS_PER_BUFFER >= 1);
                    const auto idx = buffer_pos[rq][rn] / THREADS_PER_BUFFER;
                    const auto offset = buffer_pos[rq][rn] % THREADS_PER_BUFFER;
                    const auto buffer_idx = query_offset * THREADS_PER_BUFFER + offset + idx *
                    BLOCK_SIZE;
                    */

                    // move item to the corresponding buffer
                    if (0 <= buffer_pos[rq][rn] && buffer_pos[rq][rn] < K)
                    {
                        buffer.dist[buffer_idx] = sum[rq][rn];
                        buffer.label[buffer_idx] =
                            points_offset + threadIdx.y + rn * BLOCK_POINT_DIM;
                    }

                    // check for overflow
                    overflown |= buffer_pos[rq][rn] >= static_cast<position_t>(K);

                    // decrement buffer position for the next iteration.
                    buffer_pos[rq][rn] -= static_cast<position_t>(K);
                }
            }

            // if no thread filled a buffer, we can continue without merging
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            // merge the buffers
            merge_buffers<K, BLOCK_SIZE, ITEMS_PER_THREAD, QUERIES_PER_BLOCK>(
                topk_dist, topk_label, buffer.dist, buffer.label, buffer.size);

// store radii of all lists to shared memory
#pragma unroll
            for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                const auto idx = thread_idx * ITEMS_PER_THREAD + i;
                const auto array_idx = idx / K;
                const auto element_idx = idx % K;
                if (element_idx + 1 >= K)
                {
                    buffer.dist[array_idx] = topk_dist[i];
                }
            }

            __syncthreads();

// update radii
#pragma unroll
            for (std::int32_t i = 0; i < QUERIES_PER_BLOCK; ++i)
            {
                // update radius if this thread is responsible for this query
                const auto rq = i / BLOCK_QUERY_DIM;
                const auto target_thread = i % BLOCK_QUERY_DIM;
                if (target_thread == threadIdx.x)
                {
                    radius[rq] = buffer.dist[i];
                }
            }

            __syncthreads();

// reset buffers
#pragma unroll
            for (std::int32_t i = 0; i < BUFFER_SIZE; i += BLOCK_SIZE)
            {
                const auto idx = i + thread_idx;
                if (idx < BUFFER_SIZE)
                {
                    buffer.dist[idx] = std::numeric_limits<float>::infinity();
                }
            }

            __syncthreads();
        }
    }

    // one final merge of all buffers
    merge_buffers<K, BLOCK_SIZE, ITEMS_PER_THREAD, QUERIES_PER_BLOCK>(
        topk_dist, topk_label, buffer.dist, buffer.label, buffer.size);

// store the results to global memory
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = thread_idx * ITEMS_PER_THREAD + i;
        const auto array_idx = idx / K;
        const auto element_idx = idx % K;
        const auto query_idx = array_idx + blockIdx.x * QUERIES_PER_BLOCK;

        if (query_idx < query_count && array_idx < QUERIES_PER_BLOCK)
        {
            out_dist(query_idx, element_idx) = topk_dist[i];
            out_label(query_idx, element_idx) = topk_label[i];
        }
    }
}

} // namespace

template <std::int32_t K, std::int32_t REG_QUERY_COUNT, std::int32_t REG_POINT_COUNT,
          std::int32_t BLOCK_QUERY_DIM>
void fused_kernel_runner::operator()()
{
    constexpr std::int32_t BLOCK_POINT_DIM = 128 / BLOCK_QUERY_DIM;
    constexpr std::int32_t QUERIES_PER_BLOCK = REG_QUERY_COUNT * BLOCK_QUERY_DIM;
    constexpr std::int32_t POINTS_PER_BLOCK = REG_POINT_COUNT * BLOCK_POINT_DIM;
    constexpr std::int32_t DIM_BLOCK = 1;

    // the configuration doesn't match the template parameters
    if (k != K || block_size != BLOCK_QUERY_DIM || items_per_thread[0] != REG_QUERY_COUNT ||
        items_per_thread[1] != REG_POINT_COUNT)
    {
        return; // TODO: throw an exception
    }

    const auto dim = points.size(0);
    const auto point_count = points.size(1);
    const auto query_count = queries.size(1);

    const dim3 block(BLOCK_QUERY_DIM, BLOCK_POINT_DIM);
    const dim3 grid((query_count + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK, 1);

    // compute size of the shared memory
    const std::int32_t topk_matrix_size =
        K * QUERIES_PER_BLOCK * (sizeof(float) + sizeof(std::int32_t));
    const std::int32_t query_window_size = dim * QUERIES_PER_BLOCK * sizeof(float);
    const std::int32_t point_window_size = dim * POINTS_PER_BLOCK * sizeof(float);
    const std::int32_t buffer_length_size = QUERIES_PER_BLOCK * sizeof(std::int32_t);
    const std::int32_t shm_size =
        query_window_size + point_window_size + topk_matrix_size + buffer_length_size;

    // call the kernel
    fused_kernel<REG_POINT_COUNT, REG_QUERY_COUNT, DIM_BLOCK, K, BLOCK_QUERY_DIM, BLOCK_POINT_DIM>
        <<<grid, block, shm_size>>>(queries, points, out_dist, out_label, dim, point_count,
                                    query_count);
    CUCH(cudaGetLastError());
}

#endif // FUSED_KERNEL_CUH_
