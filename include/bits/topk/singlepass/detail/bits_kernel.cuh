#ifndef BITS_TOPK_SINGLEPASS_DETAIL_BITS_KERNEL_CUH_
#define BITS_TOPK_SINGLEPASS_DETAIL_BITS_KERNEL_CUH_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"
#include "bits/topk/singlepass/bits_kernel.hpp"

#include "bits/ptx_utils.cuh"
#include "bits/topk/bitonic_sort_regs.cuh"

namespace
{

/** Broadcast the largest value in a sorted, block-wide register array @p topk
 *
 * @tparam K Size of the block-wide array @p topk
 * @tparam BLOCK_SIZE Number of threads in each thread block
 * @tparam ITEMS_PER_THREAD Number of values in each thread
 * @param shm Shared memory for the operation
 * @return the largest value in the sorted, block-wide register array @p topk
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD>
__device__ __forceinline__ float broadcast_radius(float (&topk)[ITEMS_PER_THREAD], std::size_t k)
{
    __shared__ float radius;

    assert(k > 0);
    assert(k <= BLOCK_SIZE * ITEMS_PER_THREAD);

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto element_idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (element_idx == k - 1)
        {
            radius = topk[i];
        }
    }

    __syncthreads();

    return radius;
}

/** Buffer with distance/index pairs in shared memory.
 *
 * Values (distance/label pairs) are inserted into the buffer in two steps:
 * 1) allocate buffer slot for the value (by calling `alloc()`)
 * 2) insert the value into the buffer to the allocated slot
 *
 * @tparam BUFFER_SIZE number of pairs that fit into the buffer in shared memory.
 * @tparam BLOCK_SIZE number of threads in each thread block.
 */
template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE>
struct shared_buffer
{
    enum status_t
    {
        OVERFLOW,
        NO_OVERFLOW
    };

    /** Distance of each value in the buffer.
     */
    float dist[BUFFER_SIZE];

    /** Index of each value in the buffer.
     */
    std::int32_t label[BUFFER_SIZE];

    /** Number of allocated spots for pairs in the buffer in shared memory.
     *
     * This value can be larger than `BUFFER_SIZE` if threads in a thread block try to insert
     * more values to the buffer than `BUFFER_SIZE`. In this situation, we merge the first
     * `BUFFER_SIZE` values with the top k result. The rest of the values for which we
     * allocated buffer slots will be inserted into the buffer after this operation.
     */
    std::int32_t size;

    /** Allocate buffer slots for @p batch_dist values if @p batch_dist is lower than @p radius
     *
     * @tparam ITEMS_PER_THREAD number of items per thread
     * @param[out] buffer_pos allocated position in the buffer. It can be larger than `BUFFER_SIZE`.
     *                        In that case, all threads in the thread block first call `merge()` to
     *                        clear the buffer and then insert the remaining values.
     * @param[in] batch_dist distances of loaded values
     * @param[in] radius the current kth smallest distance
     */
    template <std::size_t ITEMS_PER_THREAD>
    __device__ __forceinline__ void alloc(std::int32_t (&buffer_pos)[ITEMS_PER_THREAD],
                                          float (&batch_dist)[ITEMS_PER_THREAD], float radius)
    {
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buffer_pos[i] = batch_dist[i] < radius ? atomicAdd(&size, 1) : -1;
        }
    }

    /** Merge buffer in shared memory with the top k list (a sorted, block-wide register array
     * @p block_dist , @p topk_label )
     *
     * -# This method merges all strored values from the shared memory buffer into the top k list
     * and subtracts `BUFFER_SIZE` from the buffer's allocated size.
     * -# If the shared buffer's size is smaller than or equal to `BUFFER_SIZE`, this
     * method merges its items into the top k list and clears the buffer.
     *
     * The algorithm loads the buffer values to registers and sortes them using full Bitonic sort
     * in a descending order. It then uses a Bitonic separator to split the values to the lower and
     * upper halves in parallel. The lower half is a Bitonic sequence so it can be sorted by
     * a logarithmic number of Bitonic separator layers.
     *
     * @tparam ITEMS_PER_THREAD Number of values in each thread
     * @param topk_dist Distances in a block-wide register array
     * @param topk_label Labels in a block-wide register array
     */
    template <std::size_t ITEMS_PER_THREAD,
              typename Limit = std::integral_constant<std::int32_t, BUFFER_SIZE>>
    __device__ __forceinline__ void merge(float (&topk_dist)[ITEMS_PER_THREAD],
                                          std::int32_t (&topk_label)[ITEMS_PER_THREAD],
                                          Limit limit = {})
    {
        static_assert(BUFFER_SIZE == ITEMS_PER_THREAD * BLOCK_SIZE);

        float tmp_dist[ITEMS_PER_THREAD];
        std::int32_t tmp_label[ITEMS_PER_THREAD];

        // load buffer to registers
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = i * BLOCK_SIZE + threadIdx.x;

            tmp_dist[i] = std::numeric_limits<float>::infinity();
            __builtin_assume(idx < BUFFER_SIZE);
            if (idx < (std::int32_t)limit)
            {
                tmp_dist[i] = dist[idx];
                tmp_label[i] = label[idx];
            }
        }

        __syncthreads();

        // sort the buffer in descending order
        constexpr std::size_t SORTED_SIZE = 1;
        block_sort<SORTED_SIZE, BLOCK_SIZE, BUFFER_SIZE, ITEMS_PER_THREAD, order_t::descending>(
            tmp_dist, tmp_label, dist, label);

        // parallel split to the lower and upper halves (reversed Bitonic separator stage)
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            if (topk_dist[i] > tmp_dist[i])
            {
                topk_dist[i] = tmp_dist[i];
                topk_label[i] = tmp_label[i];
            }
        }

        // Sort the top k list. The top k list is a Bitonic sequence because it is the result of
        // a Bitonic separator. Hence, we do not have to use the full Bitonic sort.
        block_sort_bitonic<BUFFER_SIZE / 2, BLOCK_SIZE, BUFFER_SIZE>(topk_dist, topk_label, dist,
                                                                     label);

        if (threadIdx.x == 0)
        {
            size -= static_cast<std::int32_t>(BUFFER_SIZE);
        }

        __syncthreads();
    }
};

/** Load a batch of values from the input distance matrix and labels.

 * @tparam PREFETCH if true, the kernel will insert prefetch.global.L2 PTX instructions to prefetch
 the next batch.
 * @tparam ADD_NORMS if true, the kernel will add @p norms of the database vectors to @p in_dist
 distance matrix elements.
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam BATCH_SIZE number of @p in_dist elements to load for each thread.
 * @param[out] batch_dist the loaded distances from the @p in_dist matrix for each thread.
 * @param[out] batch_label the associated labels for the loaded distances. If @p in_label.data() is
 not nullptr, the kernel loads the labels from the @p in_label matrix.
 * @param[in] in_dist the input distance matrix of dimensionality @p in_dist.size(0) == gridDim.x
 (number of queries) and @p in_dist.size(1) equal to the number of database vectors.
 * @param[in] in_label the input label matrix. If @p in_label.data() is nullptr, the kernel uses
 implicit indices as labels. Otherwise, the @p in_label matrix of the same size as @p in_dist is
 used.
 * @param i the starting index of the batch to load.
 * @param label_offsets the offsets to add to the labels. This is useful for single-query problems.
 If @p label_offsets is nullptr, the kernel does not add any offset to the labels. Otherwise, size
 of @p label_offsets must be equal to @p in_dist.size(0).
 * @param norms the norms of the database vectors. If @p norms is nullptr, the kernel does not add
 any norms to the distances. Otherwise, the size of @p norms must be equal to @p in_dist.size(1).
 */
template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
__device__ __forceinline__ void
bits_kernel_load_batch(float (&batch_dist)[BATCH_SIZE], std::int32_t (&batch_label)[BATCH_SIZE],
                       array_view<float, 2> in_dist, array_view<std::int32_t, 2> in_label,
                       std::size_t i, const std::int32_t* label_offsets, const float* norms)
{
#pragma unroll
    for (std::size_t j = 0; j < BATCH_SIZE; ++j)
    {
        batch_dist[j] = std::numeric_limits<float>::infinity();

        // read the next value from input
        const auto point_idx = i + j * BLOCK_SIZE;
        if (point_idx < in_dist.size(1))
        {
            batch_dist[j] = in_dist(blockIdx.x, point_idx);

            if constexpr (ADD_NORMS)
            {
                batch_dist[j] += norms[point_idx];
            }

            batch_label[j] = !in_label.data() ? point_idx : in_label(blockIdx.x, point_idx);

            if (label_offsets != nullptr)
            {
                batch_label[j] += label_offsets[blockIdx.x];
            }
        }
    }

    // prefetch the next batch
    if constexpr (PREFETCH)
    {
#pragma unroll
        for (std::size_t j = 0; j < BATCH_SIZE; ++j)
        {
            const auto point_idx = i + BATCH_SIZE * BLOCK_SIZE + j * BLOCK_SIZE;
            if (point_idx < in_dist.size(1))
            {
                prefetch(in_dist.ptr(blockIdx.x, point_idx));
            }
        }
    }
}

/** Insert a batch of values into the shared buffer.
 *
 * Attempts to insert the loaded values into allocated buffer positions.
 * If any thread attempts to insert a value into a buffer position that is larger than the buffer
 * size, the function returns the overflow status.
 *
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam BATCH_SIZE number of values to insert for each thread.
 * @tparam BUFFER_SIZE number of value-label pairs that fit into the buffer.
 * @param batch_dist the loaded values to insert into the buffer.
 * @param batch_label the labels of the values to insert into the buffer.
 * @param shm_buffer the shared buffer to insert the values into.
 * @param buffer_pos the allocated buffer positions for the values; negative values indicate that
 * the loaded value should not be inserted.
 * @return the status of the buffer after all insertion attempts.
 */
template <std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE, std::size_t BUFFER_SIZE>
__device__ __forceinline__ shared_buffer<BUFFER_SIZE, BLOCK_SIZE>::status_t
bits_kernel_insert_batch(float (&batch_dist)[BATCH_SIZE], std::int32_t (&batch_label)[BATCH_SIZE],
                          shared_buffer<BUFFER_SIZE, BLOCK_SIZE>& shm_buffer,
                          std::int32_t (&buffer_pos)[BATCH_SIZE])
{
    bool overflown = false;

#pragma unroll
    for (std::size_t j = 0; j < BATCH_SIZE; ++j)
    {
        // try to add the loaded value to the buffer (negative position indicates no value)
        if (0 <= buffer_pos[j] && buffer_pos[j] < static_cast<std::int32_t>(BUFFER_SIZE))
        {
            shm_buffer.dist[buffer_pos[j]] = batch_dist[j];
            shm_buffer.label[buffer_pos[j]] = batch_label[j];
        }

        // check for buffer overflow
        overflown |= buffer_pos[j] >= static_cast<std::int32_t>(BUFFER_SIZE);
        buffer_pos[j] -= BUFFER_SIZE;
    }

    return __syncthreads_or(overflown) ? shm_buffer.OVERFLOW : shm_buffer.NO_OVERFLOW;
}

/** Bitonic select (bits) kernel (small k, multi-query -- one query per thread block)
 *
 * @tparam PREFETCH if true, the kernel will insert prefetch.global.L2 PTX instructions.
 * @tparam ADD_NORMS if true, the kernel will add @p norms to @p in_dist elements.
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam BATCH_SIZE number of elements to load for each thread in a single iteration.
 * @tparam K the number of values to find for each query.
 * @param[in] in_dist distance matrix.
 * @param[in] in_label label matrix (if it is nullptr, the kernel uses implicit indices as labels).
 * @param[out] out_dist top k distances for each query.
 * @param[out] out_label top k indices for each query.
 * @param[in] label_offsets offsets to add to the labels (useful for single-query problems). nullptr
 * if not needed.
 * @param[in] norms computed norms of database vectors or nullptr if @p in_dist does not require
 *                  a postprocessing.
 */
template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE,
          std::size_t K>
__global__ void __launch_bounds__(BLOCK_SIZE)
    bits_kernel(array_view<float, 2> in_dist, array_view<std::int32_t, 2> in_label,
                array_view<float, 2> out_dist, array_view<std::int32_t, 2> out_label, std::size_t k,
                const std::int32_t* label_offsets, const float* norms)
{
    // number of items stored in registers of each thread
    constexpr std::size_t ITEMS_PER_THREAD = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr std::size_t BUFFER_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    __shared__ shared_buffer<BUFFER_SIZE, BLOCK_SIZE> shm_buffer;

    shm_buffer.size = 0;

    // the top k queue
    float topk_dist[ITEMS_PER_THREAD];
    std::int32_t topk_label[ITEMS_PER_THREAD];

    assert(k > 0);
    assert(k <= K);

// load the first block
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        topk_label[i] = threadIdx.x + i * BLOCK_SIZE;
        topk_dist[i] = std::numeric_limits<float>::infinity();

        if (topk_label[i] < in_dist.size(1))
        {
            topk_dist[i] = in_dist(blockIdx.x, topk_label[i]);

            if constexpr (ADD_NORMS)
            {
                topk_dist[i] += norms[topk_label[i]];
            }

            topk_label[i] = !in_label.data() ? topk_label[i] : in_label(blockIdx.x, topk_label[i]);

            // add an optional offset to make the label unique across all "queries" in single query
            // problems
            if (label_offsets != nullptr)
            {
                topk_label[i] += label_offsets[blockIdx.x];
            }
        }
    }

    // sort the first block
    block_sort<1, BLOCK_SIZE, BUFFER_SIZE>(topk_dist, topk_label, shm_buffer.dist,
                                           shm_buffer.label);

    // initialize the radius to the kth element
    auto radius = broadcast_radius<BLOCK_SIZE>(topk_dist, k);

    // while at least one thread in the block will read a value in the next iteration
    for (auto i = BUFFER_SIZE + threadIdx.x; i < in_dist.size(1) + threadIdx.x;
         i += BATCH_SIZE * BLOCK_SIZE)
    {
        float batch_dist[BATCH_SIZE];
        std::int32_t batch_label[BATCH_SIZE];

        // load the next batch into registers
        bits_kernel_load_batch<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE>(
            batch_dist, batch_label, in_dist, in_label, i, label_offsets, norms);

        // allocate buffer positions for the loaded values if they are lower than the current radius
        std::int32_t buffer_pos[BATCH_SIZE];
        shm_buffer.alloc(buffer_pos, batch_dist, radius);

        // fill the shared buffer with the loaded values and merge into the top-k list if needed
        while (bits_kernel_insert_batch<BLOCK_SIZE, BATCH_SIZE, BUFFER_SIZE>(
                   batch_dist, batch_label, shm_buffer, buffer_pos) == shm_buffer.OVERFLOW)
        {
            // merge the buffer with the top k list
            shm_buffer.merge(topk_dist, topk_label);

            // update the radius (the kth smallest value)
            radius = broadcast_radius<BLOCK_SIZE>(topk_dist, k);
        }
    }

    shm_buffer.merge(topk_dist, topk_label, shm_buffer.size);

    // copy the values to the output
#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const std::size_t idx = threadIdx.x * ITEMS_PER_THREAD + i;

        if (idx < k)
        {
            out_dist(blockIdx.x, idx) = topk_dist[i];
            out_label(blockIdx.x, idx) = topk_label[i];
        }
    }
}

} // namespace

/** Declare an instantiation of the bits kernel.
 */
#define DECL_BITS_KERNEL(prefetch, add_norms, block_size, batch_size, k)                           \
    template void run_bits_kernel<prefetch, add_norms, block_size, batch_size, k>(                 \
        array_view<float, 2>, array_view<std::int32_t, 2>, array_view<float, 2>,                   \
        array_view<std::int32_t, 2>, std::size_t, const std::int32_t*, const float*, cudaStream_t)

template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE,
          std::size_t K>
void run_bits_kernel(array_view<float, 2> in_dist, array_view<std::int32_t, 2> in_label,
                     array_view<float, 2> out_dist, array_view<std::int32_t, 2> out_label,
                     std::size_t k, const std::int32_t* label_offsets, const float* norms,
                     cudaStream_t stream)
{
    bits_kernel<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, K>
        <<<in_dist.size(0), BLOCK_SIZE, 0, stream>>>(in_dist, in_label, out_dist, out_label, k,
                                                     label_offsets, norms);
    CUCH(cudaGetLastError());
}

#endif // BITS_TOPK_SINGLEPASS_DETAIL_BITS_KERNEL_CUH_
