#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/partial_bitonic.hpp"

#include "bits/topk/bitonic_sort.cuh"
#include "bits/topk/bitonic_sort_regs.cuh"
#include "bits/topk/bitonic_sort_static.cuh"

namespace
{

/** Load distance/label pairs from the distance matrix
 *
 * @tparam USE_WARP_SORT if true, loaded values will be pre-sorted in each warp before they're
 * stored in shared memory.
 * @tparam Layout layout of data in shared memory (see aos_layout, soa_layout)
 *
 * @param block Block in shared memory (Layout determines how are data stored in the shared memory -
 * e.g., AOS or SOA)
 * @param base_point_idx Base index of a point. Indices of distances loaded in this funtion will be
 * relative to this base.
 * @param input Distance matrix (each row contains distances from a single query)
 * @param input_count Number of points (number of columns in the @p input distance matrix)
 * @param k Size of the loaded block
 */
template <bool USE_WARP_SORT, typename Layout>
__device__ __forceinline__ void load_block(Layout block, std::int32_t base_point_idx,
                                           array_view<float, 2> input, std::size_t k)
{
    const auto input_count = input.size(1);

    // assumes k is a multiple of 32 (since the whole warp has to participate in warp_sort)
    for (std::int32_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        // load value from global memory
        auto point_idx = base_point_idx + i;
        auto dist = point_idx >= input_count ? std::numeric_limits<float>::infinity()
                                             : input(blockIdx.x, point_idx);

        // sort the loaded values in this warp
        if (USE_WARP_SORT)
        {
            warp_sort(dist, point_idx);
        }

        // store the sorted subsequences to shared memory
        block.dist(i) = dist;
        block.label(i) = point_idx;
    }

    __syncthreads();
}

/** Memory optimized version of the partial bitonic top k selection kernel
 *
 * @param[in] input Distance matrix. Each row contains distances from given query to all points in
 * the database
 * @param[out] output Top k matrix. Earch row contains the top k elements for given query.
 * @param query_count Number of query points
 * @param input_count Number of points in the database.
 * @param k Number of nearest neighbors
 * @param block_a Shared memory for the first block
 * @param block_b Shared memory for the second block
 */
template <bool USE_WARP_SORT, typename Layout>
__device__ __forceinline__ void partial_bitonic(array_view<float, 2> input,
                                                array_view<float, 2> out_dist,
                                                array_view<std::int32_t, 2> out_label,
                                                std::size_t k, Layout block_a, Layout block_b)
{
    // size of sorted subsequences for block_sort()
    constexpr std::size_t PRESORTED_SIZE = USE_WARP_SORT ? 32 : 1;

    // load and sort the first block
    load_block<USE_WARP_SORT>(block_a, 0, input, k);
    block_sort<USE_WARP_SORT, PRESORTED_SIZE>(block_a, k);

    const auto input_count = input.size(1);

    for (std::int32_t i = k; i < input_count; i += k)
    {
        // loop invariant: block_a is sorted

        // load and sort the next block
        load_block<USE_WARP_SORT>(block_b, i, input, k);
        block_sort<USE_WARP_SORT, PRESORTED_SIZE>(block_b, k);

        // merge the block with current top k
        block_merge<USE_WARP_SORT, order_t::ascending>(block_a, k, 2 * k);
        block_sort_bitonic<USE_WARP_SORT, order_t::ascending>(block_a, k / 2, k);
    }

    // copy the values to the output
    for (std::size_t i = threadIdx.x; i < k; i += blockDim.x)
    {
        out_dist(blockIdx.x, i) = block_a.dist(i);
        out_label(blockIdx.x, i) = block_a.label(i);
    }
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE, typename Layout>
__device__ __forceinline__ void
partial_bitonic(array_view<float, 2> input, array_view<float, 2> out_dist,
                array_view<std::int32_t, 2> out_label, Layout block_a, Layout block_b)
{
    // size of sorted subsequences for block_sort()
    constexpr std::size_t PRESORTED_SIZE = USE_WARP_SORT ? 32 : 1;

    // load and sort the first block
    load_block<USE_WARP_SORT>(block_a, 0, input, K);
    block_sort<USE_WARP_SORT, PRESORTED_SIZE, K, BLOCK_SIZE>(block_a);

    const auto input_count = input.size(1);

    for (std::int32_t i = K; i < input_count; i += K)
    {
        // loop invariant: block_a is sorted

        // load and sort the next block
        load_block<USE_WARP_SORT>(block_b, i, input, K);
        block_sort<USE_WARP_SORT, PRESORTED_SIZE, K, BLOCK_SIZE>(block_b);

        // merge the block with current top k
        block_merge<USE_WARP_SORT, K, 2 * K, BLOCK_SIZE, Layout>(block_a);
        block_sort_bitonic<USE_WARP_SORT, K / 2, K, BLOCK_SIZE, Layout>(block_a);
    }

    // copy the values to the output
    for (std::size_t i = threadIdx.x; i < K; i += blockDim.x)
    {
        out_dist(blockIdx.x, i) = block_a.dist(i);
        out_label(blockIdx.x, i) = block_a.label(i);
    }
}

template <bool USE_WARP_SORT>
__global__ void partial_bitonic_aos(array_view<float, 2> input, array_view<float, 2> out_dist,
                                    array_view<std::int32_t, 2> out_label, std::size_t k)
{
    extern __shared__ knn::pair_t values[];

    // split shared memory to two blocks
    const auto block_a = aos_layout{values};
    const auto block_b = aos_layout{values + k};

    partial_bitonic<USE_WARP_SORT>(input, out_dist, out_label, k, block_a, block_b);
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE>
__global__ void partial_bitonic_aos(array_view<float, 2> input, array_view<float, 2> out_dist,
                                    array_view<std::int32_t, 2> out_label)
{
    extern __shared__ knn::pair_t values[];

    // split shared memory to two blocks
    const auto block_a = aos_layout{values};
    const auto block_b = aos_layout{values + K};

    partial_bitonic<USE_WARP_SORT, K, BLOCK_SIZE>(input, out_dist, out_label, block_a, block_b);
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE>
__global__ void partial_bitonic_soa(array_view<float, 2> input, array_view<float, 2> out_dist,
                                    array_view<std::int32_t, 2> out_label)
{
    constexpr std::size_t PRESORTED_SIZE = USE_WARP_SORT ? 32 : 1;

    extern __shared__ std::uint8_t shm[];

    // split the shared memory to two blocks A and B
    const auto shm_dist_a = reinterpret_cast<float*>(shm);
    const auto shm_dist_b = reinterpret_cast<float*>(shm + K * sizeof(float));

    const auto shm_labels_a = reinterpret_cast<std::int32_t*>(shm + 2 * K * sizeof(float));
    const auto shm_labels_b =
        reinterpret_cast<std::int32_t*>(shm + 2 * K * sizeof(float) + K * sizeof(std::int32_t));

    // load and sort the first block
    load_block<USE_WARP_SORT>(soa_layout{shm_dist_a, shm_labels_a}, 0, input, K);
    block_sort<USE_WARP_SORT, PRESORTED_SIZE, K, BLOCK_SIZE>(soa_layout{shm_dist_a, shm_labels_a});

    const auto input_count = input.size(1);

    for (std::size_t block_offset = K; block_offset < input_count; block_offset += K)
    {
        // loop invariant: block_a is sorted

        // load and sort the next block from global memory
        load_block<USE_WARP_SORT>(soa_layout{shm_dist_b, shm_labels_b}, block_offset, input, K);
        block_sort<USE_WARP_SORT, PRESORTED_SIZE, K, BLOCK_SIZE>(
            soa_layout{shm_dist_b, shm_labels_b});

        block_merge<USE_WARP_SORT, K, 2 * K, BLOCK_SIZE, soa_layout>(
            soa_layout{shm_dist_a, shm_labels_a});
        block_sort_bitonic<USE_WARP_SORT, K / 2, K, BLOCK_SIZE, soa_layout>(
            soa_layout{shm_dist_a, shm_labels_a});
    }

    // store the result back to global memory
    for (std::size_t i = threadIdx.x; i < K; i += blockDim.x)
    {
        out_dist(blockIdx.x, i) = shm_dist_a[i];
        out_label(blockIdx.x, i) = shm_labels_a[i];
    }
}

template <std::size_t K, std::size_t BLOCK_SIZE>
__global__ void partial_bitonic_regs_kernel(array_view<float, 2> input,
                                            array_view<float, 2> out_dist,
                                            array_view<std::int32_t, 2> out_label)
{
    constexpr std::size_t ITEMS_PER_THREAD = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // the lowest multiple of BLOCK_SIZE greater or equal to K
    // (can be evenly divided among the threads)
    constexpr std::size_t BUFFER_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    // split the shared memory to two blocks A and B
    __shared__ float shm_dist[BLOCK_SIZE * 2];
    __shared__ std::int32_t shm_label[BLOCK_SIZE * 2];

    const auto input_count = input.size(1);

    // thread local arrays
    float dist[ITEMS_PER_THREAD];
    std::int32_t label[ITEMS_PER_THREAD];

// load and sort the first block
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        label[i] = i * BLOCK_SIZE + threadIdx.x;
        dist[i] = label[i] >= input_count ? std::numeric_limits<float>::infinity()
                                          : input(blockIdx.x, label[i]);
    }

    block_sort<1, BLOCK_SIZE, BUFFER_SIZE>(dist, label, shm_dist, shm_label);

    for (std::size_t block_offset = BUFFER_SIZE; block_offset < input_count;
         block_offset += BUFFER_SIZE)
    {
        // loop invariant: block_a is sorted

        // load and sort the next block from global memory
        float next_dist[ITEMS_PER_THREAD];
        std::int32_t next_label[ITEMS_PER_THREAD];

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            next_label[i] = block_offset + i * BLOCK_SIZE + threadIdx.x;
            next_dist[i] = next_label[i] >= input_count ? std::numeric_limits<float>::infinity()
                                                        : input(blockIdx.x, next_label[i]);
        }

        block_sort<1, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD, order_t::descending>(
            next_dist, next_label, shm_dist, shm_label);

        // dist is sorted in ascending order, next_dist is sorted in descending order
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            if (dist[i] > next_dist[i])
            {
                dist[i] = next_dist[i];
                label[i] = next_label[i];
            }
        }

        block_sort_bitonic<BUFFER_SIZE / 2, BLOCK_SIZE, BUFFER_SIZE>(dist, label, shm_dist,
                                                                     shm_label);
    }

// store the result back to global memory
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (idx < K)
        {
            out_dist(blockIdx.x, idx) = dist[i];
            out_label(blockIdx.x, idx) = label[i];
        }
    }
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE>
void partial_bitonic_soa_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label)
{
    partial_bitonic_soa<USE_WARP_SORT, K, BLOCK_SIZE>
        <<<input.size(0), BLOCK_SIZE,
           2 * K * (sizeof(float) + sizeof(std::int32_t)) + sizeof(std::int32_t)>>>(input, out_dist,
                                                                                    out_label);
}

template <bool USE_WARP_SORT, std::size_t K, std::size_t BLOCK_SIZE>
void partial_bitonic_aos_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label)
{
    partial_bitonic_aos<USE_WARP_SORT, K, BLOCK_SIZE>
        <<<input.size(0), BLOCK_SIZE,
           2 * K * (sizeof(float) + sizeof(std::int32_t)) + sizeof(std::int32_t)>>>(input, out_dist,
                                                                                    out_label);
}

template <std::size_t BLOCK_SIZE, std::size_t K>
void partial_bitonic_regs_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                              array_view<std::int32_t, 2> out_label)
{
    partial_bitonic_regs_kernel<K, BLOCK_SIZE>
        <<<input.size(0), BLOCK_SIZE>>>(input, out_dist, out_label);
}

template <bool USE_WARP_SORT, std::size_t BLOCK_SIZE>
void partial_bitonic_soa_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label, std::size_t k)
{
    if (k == 32)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 32, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 64)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 64, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 128)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 128, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 256)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 256, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 512)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 512, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 1024)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 1024, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 2048)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 2048, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else
    {
        throw std::runtime_error("Unsupported k value");
    }
}

template <bool USE_WARP_SORT, std::size_t BLOCK_SIZE>
void partial_bitonic_aos_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label, std::size_t k)
{
    if (k == 32)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 32, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 64)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 64, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 128)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 128, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 256)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 256, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 512)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 512, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 1024)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 1024, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else if (k == 2048)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 2048, BLOCK_SIZE>(input, out_dist, out_label);
    }
    else
    {
        throw std::runtime_error("Unsupported k value");
    }
}

template <std::size_t BLOCK_SIZE>
void partial_bitonic_regs_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                              array_view<std::int32_t, 2> out_label, std::size_t k)
{
    if (k == 32)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 32>(input, out_dist, out_label);
    }
    else if (k == 64)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 64>(input, out_dist, out_label);
    }
    else if (k == 128)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 128>(input, out_dist, out_label);
    }
    else if (k == 256)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 256>(input, out_dist, out_label);
    }
    else if (k == 512)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 512>(input, out_dist, out_label);
    }
    else if (k == 1024)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 1024>(input, out_dist, out_label);
    }
    else if (k == 2048)
    {
        partial_bitonic_regs_run<BLOCK_SIZE, 2048>(input, out_dist, out_label);
    }
    else
    {
        throw std::runtime_error("Unsupported k value");
    }
}

template <bool USE_WARP_SORT>
void partial_bitonic_soa_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label, std::size_t block_size,
                             std::size_t k)
{
    if (block_size == 64)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 64>(input, out_dist, out_label, k);
    }
    else if (block_size == 128)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 128>(input, out_dist, out_label, k);
    }
    else if (block_size == 256)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 256>(input, out_dist, out_label, k);
    }
    else if (block_size == 512)
    {
        partial_bitonic_soa_run<USE_WARP_SORT, 512>(input, out_dist, out_label, k);
    }
    else
    {
        throw std::runtime_error("Unsupported block size");
    }
}

template <bool USE_WARP_SORT>
void partial_bitonic_aos_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                             array_view<std::int32_t, 2> out_label, std::size_t block_size,
                             std::size_t k)
{
    if (block_size == 64)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 64>(input, out_dist, out_label, k);
    }
    else if (block_size == 128)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 128>(input, out_dist, out_label, k);
    }
    else if (block_size == 256)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 256>(input, out_dist, out_label, k);
    }
    else if (block_size == 512)
    {
        partial_bitonic_aos_run<USE_WARP_SORT, 512>(input, out_dist, out_label, k);
    }
    else
    {
        throw std::runtime_error("Unsupported block size");
    }
}

void partial_bitonic_regs_run(array_view<float, 2> input, array_view<float, 2> out_dist,
                              array_view<std::int32_t, 2> out_label, std::size_t block_size,
                              std::size_t k)
{
    if (block_size == 64)
    {
        partial_bitonic_regs_run<64>(input, out_dist, out_label, k);
    }
    else if (block_size == 128)
    {
        partial_bitonic_regs_run<128>(input, out_dist, out_label, k);
    }
    else if (block_size == 256)
    {
        partial_bitonic_regs_run<256>(input, out_dist, out_label, k);
    }
    else if (block_size == 512)
    {
        partial_bitonic_regs_run<512>(input, out_dist, out_label, k);
    }
    else
    {
        throw std::runtime_error("Unsupported block size");
    }
}

} // namespace

void partial_bitonic::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();

    partial_bitonic_aos<false>
        <<<block_count, selection_block_size(),
           2 * k() * (sizeof(float) + sizeof(std::int32_t)) + sizeof(std::int32_t)>>>(
            in_dist_gpu(), out_dist_gpu(), out_label_gpu(), k());

    cuda_stream::make_default().sync();
}

void partial_bitonic_warp::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();

    partial_bitonic_aos<true>
        <<<block_count, selection_block_size(),
           2 * k() * (sizeof(float) + sizeof(std::int32_t)) + sizeof(std::int32_t)>>>(
            in_dist_gpu(), out_dist_gpu(), out_label_gpu(), k());

    cuda_stream::make_default().sync();
}

void partial_bitonic_warp_static::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();
    const auto block_size = selection_block_size();

    partial_bitonic_aos_run<true>(dist, out_dist, out_label, block_size, k());

    cuda_stream::make_default().sync();
}

void partial_bitonic_arrays::selection()
{
    cuda_knn::selection();

    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();
    const auto block_size = selection_block_size();

    partial_bitonic_soa_run<true>(dist, out_dist, out_label, block_size, k());

    cuda_stream::make_default().sync();
}

void partial_bitonic_regs::selection()
{
    const auto block_count = query_count();
    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();
    const auto block_size = selection_block_size();

    partial_bitonic_regs_run(dist, out_dist, out_label, block_size, k());

    cuda_stream::make_default().sync();
}
