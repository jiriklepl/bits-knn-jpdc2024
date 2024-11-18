#ifndef BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_KERNEL_STRUCTS_CUH_
#define BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_KERNEL_STRUCTS_CUH_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "bits/topk/bitonic_sort_regs.cuh"

/** Distance/label buffers in shared memory.
 *
 * This structure implements a parallel merge operation of several small distance/label buffers
 * with top k arrays.
 *
 * @tparam K size of each buffer/top k array.
 * @tparam BLOCK_QUERY_DIM number of threads in a thread block along the query dimension.
 * @tparam BLOCK_DB_DIM number of threads in a thread block along the DB dimension.
 * @tparam ITEMS_PER_THREAD number of items per thread to store values from all buffers in
 * registers of a thread block.
 * @tparam ARRAY_COUNT number of buffers/top k arrays.
 */
template <std::int32_t K, std::int32_t BLOCK_QUERY_DIM, std::int32_t BLOCK_DB_DIM,
          std::int32_t ITEMS_PER_THREAD, std::int32_t ARRAY_COUNT>
struct multi_buffer
{
    // total number of threads in a thread block
    static constexpr std::int32_t BLOCK_SIZE = BLOCK_QUERY_DIM * BLOCK_DB_DIM;
    // combined size of all buffers
    static constexpr std::int32_t TOTAL_CAPACITY = ARRAY_COUNT * K;

    struct multi_buffer_t
    {
        // keys in the buffer
        float keys[TOTAL_CAPACITY];
        // values in the buffer
        std::int32_t values[TOTAL_CAPACITY];
    };

    struct tmp_storage_t
    {
        // size of each buffer (number of valid elements)
        std::int32_t buffer_size[ARRAY_COUNT];

        union
        {
            // buffer for each array
            multi_buffer_t buffer;

            // memory for sorting
            struct
            {
                float keys[2 * BLOCK_SIZE];
                std::int32_t values[2 * BLOCK_SIZE];
            } sort;
        };
    };

    tmp_storage_t& tmp_storage;

    __device__ __forceinline__ explicit multi_buffer(tmp_storage_t& shm) : tmp_storage(shm)
    {
        const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

        // initialize buffer size
        for (std::int32_t i = thread_idx; i < ARRAY_COUNT; i += BLOCK_SIZE)
        {
            tmp_storage.buffer_size[i] = 0;
        }

        // reset buffer
        for (std::int32_t i = thread_idx; i < TOTAL_CAPACITY; i += BLOCK_SIZE)
        {
            tmp_storage.buffer.keys[i] = std::numeric_limits<float>::infinity();
        }
    }

    /** Allocate buffer positions for all distances in @p dist which are smaller than @p radius
     *
     * @tparam QUERY_REG number of registers along the query dimension.
     * @tparam DB_REG number of registers along the DB dimension.
     * @param[in] dist computed distances.
     * @param[in] radius the kth smallest distance for each buffer.
     * @param[out] buffer_pos allocated buffer index for given distance or -1.
     * @param[in] num_queries number of valid queries.
     * @param[in] num_db number of valid DB vectors.
     */
    template <std::int32_t QUERY_REG, std::int32_t DB_REG>
    __device__ __forceinline__ void alloc(float (&dist)[QUERY_REG][DB_REG],
                                          float (&radius)[QUERY_REG],
                                          std::int32_t (&buffer_pos)[QUERY_REG][DB_REG],
                                          std::int32_t num_queries, std::int32_t num_db)
    {
#pragma unroll
        for (std::int32_t q = 0; q < QUERY_REG; ++q)
        {
#pragma unroll
            for (std::int32_t i = 0; i < DB_REG; ++i)
            {
                const std::int32_t query_idx = threadIdx.x + q * BLOCK_QUERY_DIM;
                const std::int32_t db_idx = threadIdx.y + i * BLOCK_DB_DIM;
                if (query_idx < num_queries && db_idx < num_db && dist[q][i] < radius[q])
                {
                    buffer_pos[q][i] = atomicAdd(tmp_storage.buffer_size + query_idx, 1);
                }
                else
                {
                    buffer_pos[q][i] = -1; // do not insert this value into the buffer
                }
            }
        }
    }

    /** Biven buffer positions allocated by `alloc()`, insert distance/label pairs into the buffer.
     *
     * If a buffer overflows, all buffers are sorted and merged with the top k lists stored in one
     * block-wide register array @p topk_dist and @p topk_label
     *
     * @param[in] dist computed distances.
     * @param[in] buffer_pos positions allocated by `alloc()`.
     * @param[in,out] topk_dist top k arrays of distances stored as one block-wide register array.
     * @param[in,out] topk_label top k arrays of labels stored as one block-wide register array.
     * @param[in,out] radius the kth smallest distance for each query. Radii are updated when
     * the buffers are merged.
     * @param label_offset value added to all labels.
     */
    template <std::int32_t QUERY_REG, std::int32_t DB_REG>
    __device__ __forceinline__ void
    insert(float (&dist)[QUERY_REG][DB_REG], std::int32_t (&buffer_pos)[QUERY_REG][DB_REG],
           float (&topk_dist)[ITEMS_PER_THREAD], std::int32_t (&topk_label)[ITEMS_PER_THREAD],
           float (&radius)[QUERY_REG], std::int32_t label_offset)
    {
        // merge buffers if necessary
        for (;;)
        {
            bool overflown = false;

#pragma unroll
            for (std::int32_t q = 0; q < QUERY_REG; ++q)
            {
#pragma unroll
                for (std::int32_t i = 0; i < DB_REG; ++i)
                {
                    const std::int32_t query_idx = threadIdx.x + q * BLOCK_QUERY_DIM;
                    const std::int32_t db_idx = threadIdx.y + i * BLOCK_DB_DIM;
                    // linear access with conflicts in buffer.merge()
                    const auto buffer_idx = query_idx * K + buffer_pos[q][i];

                    // move item to the corresponding buffer
                    if (0 <= buffer_pos[q][i] && buffer_pos[q][i] < K)
                    {
                        tmp_storage.buffer.keys[buffer_idx] = dist[q][i];
                        tmp_storage.buffer.values[buffer_idx] = label_offset + db_idx;
                    }

                    // check for overflow
                    overflown |= buffer_pos[q][i] >= K;

                    // decrement buffer position for the next iteration.
                    buffer_pos[q][i] -= K;
                }
            }

            // if no thread filled a buffer, we can continue without merging
            if (!__syncthreads_or(overflown))
            {
                break;
            }

            merge(topk_dist, topk_label);
            broadcast_radii(topk_dist, radius);
        }
    }

    /** Merge buffers in shared memory with a sorted top k block-wide register arrays @p dist and
     * @p label
     *
     * @param dist `ARRAY_COUNT` top k arrays of distances treated as one block-wide register array.
     * @param label `ARRAY_COUNT` top k arrays of labels treated as one block-wide register array.
     */
    __device__ __forceinline__ void merge(float (&dist)[ITEMS_PER_THREAD],
                                          std::int32_t (&label)[ITEMS_PER_THREAD])
    {
        // static_assert(ITEMS_PER_THREAD <= K);

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
            if (buffer_idx < VALUE_COUNT)
            {
                buffer_dist[i] = tmp_storage.buffer.keys[buffer_idx];
                buffer_label[i] = tmp_storage.buffer.values[buffer_idx];
            }
        }

        __syncthreads();

        // sort all buffers
        block_sort_partial<1, K, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, order_t::descending>(
            buffer_dist, buffer_label, tmp_storage.sort.keys, tmp_storage.sort.values);

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
        block_sort_bitonic<K / 2, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD>(
            dist, label, tmp_storage.sort.keys, tmp_storage.sort.values);

        // reset buffer size
        for (std::int32_t i = thread_idx; i < ARRAY_COUNT; i += BLOCK_SIZE)
        {
            tmp_storage.buffer_size[i] = std::max<std::int32_t>(tmp_storage.buffer_size[i] - K, 0);
        }

        __syncthreads();
    }

    /** Broadcast the current radius (the kth smallest distance) of all queries to all threads.
     *
     * @param[in] topk_dist block-wide register array which contains the top k lists of all
     * queries of this thread block as a one thread block-wide array in the blocked arrangement.
     * @param[out] radius the kth smallest distance for each query of this thread block.
     */
    template <std::int32_t QUERY_REG>
    __device__ __forceinline__ void broadcast_radii(float (&topk_dist)[ITEMS_PER_THREAD],
                                                    float (&radius)[QUERY_REG])
    {
        const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

// store radii of all lists to shared memory
#pragma unroll
        for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = thread_idx * ITEMS_PER_THREAD + i;
            const auto array_idx = idx / K;
            const auto element_idx = idx % K;
            if (element_idx + 1 >= K)
            {
                tmp_storage.buffer.keys[array_idx] = topk_dist[i];
            }
        }

        __syncthreads();

// update radii
#pragma unroll
        for (std::int32_t i = 0; i < QUERY_REG; ++i)
        {
            if (threadIdx.x + i * BLOCK_QUERY_DIM < ARRAY_COUNT)
            {
                radius[i] = tmp_storage.buffer.keys[threadIdx.x + i * BLOCK_QUERY_DIM];
            }
        }

        __syncthreads();

        reset();
    }

    /** Reset buffer keys to infinity.
     */
    __device__ __forceinline__ void reset()
    {
        const auto thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

// reset buffers
#pragma unroll
        for (std::int32_t i = 0; i < TOTAL_CAPACITY; i += BLOCK_SIZE)
        {
            const auto idx = i + thread_idx;
            if (idx < TOTAL_CAPACITY)
            {
                tmp_storage.buffer.keys[idx] = std::numeric_limits<float>::infinity();
            }
        }

        __syncthreads();
    }
};

/** Class which computes Euclidean distance.
 *
 * @tparam QUERY_REG number of registers per thread dedicated for queries.
 * @tparam DB_REG number of registers per thread dedicated for DB vectors.
 * @tparam DIM_REG number of dimensions cached in shared memory per tile.
 * @tparam BLOCK_QUERY_DIM number of threads in a thread block along the query dimension.
 * @tparam BLOCK_POINT_DIM number of threads in a thread block along the DB vector dimension.
 */
template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG,
          std::int32_t BLOCK_QUERY_DIM, std::int32_t BLOCK_POINT_DIM, std::int32_t DIM_MULT>
struct distance_computation
{
    // number of queries processed by each thread block
    static constexpr std::int32_t QUERY_TILE = BLOCK_QUERY_DIM * QUERY_REG;
    // size of the database vector window (number of database vectors)
    static constexpr std::int32_t DB_TILE = BLOCK_POINT_DIM * DB_REG;
    // size of each tile along the common dimension
    static constexpr std::int32_t DIM_TILE = DIM_MULT * DIM_REG;
    // number of threads in a thread block
    static constexpr std::int32_t BLOCK_SIZE = BLOCK_QUERY_DIM * BLOCK_POINT_DIM;
    // number of vector components loaded per thread for each query tile
    static constexpr std::int32_t QUERY_ITEMS_PER_THREAD =
        (QUERY_TILE * DIM_TILE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // number of vector components loaded per thread fore each DB tile per dimension
    static constexpr std::int32_t DB_LOADS_PER_THREAD = (DB_TILE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /** Shared memory used by this component.
     */
    struct tmp_storage_t
    {
        // shared memory for a tile of queries assigned to this thread block
        float queries[DIM_TILE][QUERY_TILE];
        // shared memory for a tile of the current window of the database vecotrs
        float db[DIM_TILE][DB_TILE];
    };

    tmp_storage_t& tmp_storage;
    // registers for pre-loading values from global memory
    float preload_db[DIM_TILE][DB_LOADS_PER_THREAD];
    // registers for pre-loading values from global memory
    float preload_queries[QUERY_ITEMS_PER_THREAD];

    /** Type of memory movement within a tile to perform.
     */
    enum class load
    {
        /** Load tile data from global memory -> shared memory.
         */
        shm,

        /** Load tile data from global memory -> registers.
         */
        reg,

        /** Store tile data from registers -> shared memory.
         */
        move
    };

    __device__ __forceinline__ explicit distance_computation(tmp_storage_t& shm) : tmp_storage(shm)
    {
    }

    /** Move a tile of DB vectors.
     *
     * @tparam op move operation which defines the source and the destination.
     * @param data DB vector data in global memory.
     * @param stride stride of @p data to get to the next row.
     * @param num_dim number of valid dimensions in this tile.
     * @param num_vectors number of valid vectors in this tile.
     */
    template <load op>
    __device__ __forceinline__ void db_tile(const float* data, std::int32_t stride,
                                            std::int32_t num_dim, std::int32_t num_vectors)
    {
        const std::int32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
        for (std::int32_t dim = 0; dim < DIM_TILE; ++dim)
        {
            std::int32_t idx = thread_idx;
#pragma unroll
            for (std::int32_t i = 0; i < DB_LOADS_PER_THREAD; ++i)
            {
                if (dim < num_dim && idx < num_vectors)
                {
                    if constexpr (op == load::shm)
                    {
                        tmp_storage.db[dim][idx] = data[idx];
                    }
                    else if constexpr (op == load::reg)
                    {
                        preload_db[dim][i] = data[idx];
                    }
                    else if constexpr (op == load::move)
                    {
                        tmp_storage.db[dim][idx] = preload_db[dim][i];
                    }
                }
                idx += BLOCK_SIZE;
            }
            data += stride;
        }
    }

    /** Move a tile of query vectors.
     *
     * @tparam op move operation which defines the source and the destination.
     * @param data query vector data in global memory.
     * @param stride stride of @p data to get to the next row.
     * @param num_dim number of valid dimensions in this tile.
     * @param num_vectors number of valid vectors in this tile.
     */
    template <load op>
    __device__ __forceinline__ void query_tile(const float* data, std::int32_t stride,
                                               std::int32_t num_dim, std::int32_t num_vectors)
    {
        const std::int32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
        for (std::int32_t i = 0; i < QUERY_ITEMS_PER_THREAD; ++i)
        {
            const std::int32_t linear_idx = thread_idx + i * BLOCK_SIZE;
            const std::int32_t dim = linear_idx / QUERY_TILE;
            const std::int32_t idx = linear_idx % QUERY_TILE;
            if (dim < num_dim && idx < num_vectors)
            {
                if constexpr (op == load::shm)
                {
                    tmp_storage.queries[dim][idx] = data[dim * stride + idx];
                }
                else if constexpr (op == load::reg)
                {
                    preload_queries[i] = data[dim * stride + idx];
                }
                else if constexpr (op == load::move)
                {
                    tmp_storage.queries[dim][idx] = preload_queries[i];
                }
            }
        }
    }

    /** Compute partial distances from a tile of size `DIM_REG * QUERIES_PER_BLOCK` and
     * `DIM_REG * DB_TILE`.
     *
     * @param[in,out] dist distance accumulator.
     * @param[in,out] db_norm accumulator of squared DB vector norms.
     * @param[in] num_dim number of valid dimensions.
     */
    __device__ __forceinline__ void compute_tile(float (&dist)[QUERY_REG][DB_REG],
                                                 float (&db_norm)[DB_REG], std::int32_t num_dim)
    {
        float reg_queries[DIM_REG][QUERY_REG];
        float reg_db[DIM_REG][DB_REG];

#pragma unroll
        for (std::int32_t dd = 0; dd < DIM_TILE; dd += DIM_REG)
        {
            // load DB vectors to registers
#pragma unroll
            for (std::int32_t d = 0; d < DIM_REG; ++d)
            {
                const float* sd = tmp_storage.db[dd + d] + threadIdx.y;

                // load DB vectors shm -> registers
#pragma unroll
                for (std::int32_t i = 0; i < DB_REG; ++i)
                {
                    reg_db[d][i] = *sd;
                    sd += BLOCK_POINT_DIM;
                }
            }

            // load queries to registers
#pragma unroll
            for (std::int32_t d = 0; d < DIM_REG; ++d)
            {
                const float* sq = tmp_storage.queries[dd + d] + threadIdx.x;

                // load queries shm -> registers
#pragma unroll
                for (std::int32_t q = 0; q < QUERY_REG; ++q)
                {
                    reg_queries[d][q] = *sq;
                    sq += BLOCK_QUERY_DIM;
                }
            }

            // compute
#pragma unroll
            for (std::int32_t d = 0; d < DIM_REG; ++d)
            {
                if (dd + d < num_dim)
                {
                    // compute DB vector norms
#pragma unroll
                    for (std::int32_t i = 0; i < DB_REG; ++i)
                    {
                        db_norm[i] += reg_db[d][i] * reg_db[d][i];
                    }

                    // compute the dot product of DB and query vectors
#pragma unroll
                    for (std::int32_t i = 0; i < DB_REG; ++i)
                    {
#pragma unroll
                        for (std::int32_t q = 0; q < QUERY_REG; ++q)
                        {
                            dist[q][i] += reg_queries[d][q] * reg_db[d][i];
                        }
                    }
                }
            }
        }
    }

    /** Compute distance from `QUERIES_PER_BLOCK` queries to `DB_TILE` database vectors.
     *
     * @param[in] queries pointer to the first query vector.
     * @param[in] db pointer to the first database vector.
     * @param[in] queries_stride stride used to get to the next row in @p queries
     * @param[in] db_stride stride used to get to the next row in @p db
     * @param[in] dim dimension of vectors.
     * @param[in] num_queries number of valid queries in the next tile.
     * @param[in] num_db number of valid database vectors in the next tile.
     * @param[in,out] dist distance accumulator.
     */
    __device__ __forceinline__ void compute(const float* __restrict__ queries,
                                            const float* __restrict__ db,
                                            std::int32_t queries_stride, std::int32_t db_stride,
                                            std::int32_t dim, std::int32_t num_queries,
                                            std::int32_t num_db, float (&dist)[QUERY_REG][DB_REG])
    {
        // norms of DB vectors
        float db_norm[DB_REG];
#pragma unroll
        for (std::int32_t i = 0; i < DB_REG; ++i)
        {
            db_norm[i] = 0;

#pragma unroll
            for (std::int32_t q = 0; q < QUERY_REG; ++q)
            {
                dist[q][i] = 0;
            }
        }

        assert(dim < DIM_TILE);

        // load the first DB tile (global memory -> shared memory)
        db_tile<load::shm>(db, db_stride, DIM_TILE, num_db);
        // load the first query tile (global memory/L2 cache -> shared memory)
        query_tile<load::shm>(queries, queries_stride, DIM_TILE, num_queries);

        __syncthreads();

        std::int32_t d = DIM_TILE;
        for (; d + DIM_TILE <= dim; d += DIM_TILE)
        {
            queries += DIM_TILE * queries_stride;
            db += DIM_TILE * db_stride;

            // load the next query tile (global memory -> registers)
            query_tile<load::reg>(queries, queries_stride, DIM_TILE, num_queries);
            // preload the next DB tile (global memory -> registers)
            db_tile<load::reg>(db, db_stride, DIM_TILE, num_db);
            // compute partial distances from the tiles loaded in shared memory
            compute_tile(dist, db_norm, DIM_TILE);

            __syncthreads();

            // set the query tile to the next tile that is loaded already (registers -> shared
            // memory)
            query_tile<load::move>(queries, queries_stride, DIM_TILE, num_queries);
            // set the DB tile to the next tile that is loaded already (registers -> shared memory)
            db_tile<load::move>(db, db_stride, DIM_TILE, num_db);

            __syncthreads();
        }

        // the last iteration with bound checks for dimensions
        if (d < dim)
        {
            queries += DIM_TILE * queries_stride;
            db += DIM_TILE * db_stride;

            // load the next query tile (global memory -> registers)
            query_tile<load::reg>(queries, queries_stride, dim - d, num_queries);
            // preload the next DB tile (global memory -> registers)
            db_tile<load::reg>(db, db_stride, dim - d, num_db);
            // Compute partial distances from the tiles loaded in shared memory.
            // The tile in shared memory is from the previous iteration which loaded a full tile so
            // we do not have to check bounds.
            compute_tile(dist, db_norm, DIM_TILE);

            __syncthreads();

            // set the query tile to the next tile that is loaded already (registers -> shared
            // memory)
            query_tile<load::move>(queries, queries_stride, dim - d, num_queries);
            // set the DB tile to the next tile that is loaded already (registers -> shared memory)
            db_tile<load::move>(db, db_stride, dim - d, num_db);

            __syncthreads();

            // compute the last tile
            compute_tile(dist, db_norm, dim - d);
        }
        else // compute the last tile
        {
            compute_tile(dist, db_norm, DIM_TILE);
        }

        // finish the computation
#pragma unroll
        for (std::int32_t i = 0; i < DB_REG; ++i)
        {
#pragma unroll
            for (std::int32_t q = 0; q < QUERY_REG; ++q)
            {
                dist[q][i] = db_norm[i] - 2 * dist[q][i];
            }
        }
    }
};

/** Fused Euclidean distance computation kernel and top k selection kernel which does not load the
 * whole vectors to shared memory.
 *
 * Input matrices are split into tiles. The tiles do not necessarily cover all dimensions like in
 * the `fused_regs_kernel`. Query vectors are thus loaded from global memory several times, but
 * these requests can usually be satisfied from the cache.
 *
 * The memory management is inspired by the MAGMA GEMM kernel. In each iteration, we load the next
 * tile to registers and then we run the distance computation on data previously loaded to shared
 * memory. Smaller subtiles are loaded from shared memory to registers. The core computation works
 * with these register subtiles.
 *
 * The final computation computes the dot product between all query and DB vectors. It also
 * computes the norms of the DB vectors. When we have the dot products and norms computed, the
 * kernel finishes the distances by computing `norm(x)^2 - 2 * dot(x, y)` where `x` is a DB vector
 * and `y` is a query vector (i.e., it does not compute actual Euclidean distances; however, simply
 * adding `norm(y)^2` to the computed quantity produces squared Euclidean distances). This
 * modification possibly uses less registers and has less FFMA instructions in the critical loop.
 * It does not affect the order the top k results.
 *
 * @tparam QUERY_REG number of registers along the query dimension.
 * @tparam DB_REG number of registers along the DB dimension.
 * @tparam DIM_REG number of registers for vector dimension.
 * @tparam BLOCK_QUERY_DIM thread block size along the query dimension.
 * @tparam BLOCK_DB_DIM thread block size along the DB dimension.
 * @tparam DIM_MULT `DIM_REG` will be multiplied by this constant to determine the number of
 * vector dimensions loaded in shared memory for each tile.
 * @tparam K size of the output (must be a power-of-two).
 */
template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG,
          std::int32_t BLOCK_QUERY_DIM, std::int32_t BLOCK_DB_DIM, std::int32_t DIM_MULT,
          std::int32_t K>
struct fused_cache_kernel
{
    // minimal number of thread blocks per SM to limit register usage (to achieve a higher
    // occupancy)
    static constexpr std::int32_t MIN_BLOCKS_PER_SM = 2;
    // total number of threads in a thread block
    static constexpr std::int32_t BLOCK_SIZE = BLOCK_QUERY_DIM * BLOCK_DB_DIM;
    // number of queries processed in a thread block
    static constexpr std::int32_t QUERY_TILE = BLOCK_QUERY_DIM * QUERY_REG;
    // number of database vectors in a window
    static constexpr std::int32_t DB_TILE = BLOCK_DB_DIM * DB_REG;
    // number of items per thread for top k arrays and buffers
    static constexpr std::int32_t ITEMS_PER_THREAD = (QUERY_TILE * K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    using dist_t =
        distance_computation<QUERY_REG, DB_REG, DIM_REG, BLOCK_QUERY_DIM, BLOCK_DB_DIM, DIM_MULT>;
    using buffer_t = multi_buffer<K, BLOCK_QUERY_DIM, BLOCK_DB_DIM, ITEMS_PER_THREAD, QUERY_TILE>;

    /** Shared memory used by this kernel
     */
    struct tmp_storage_t
    {
        typename dist_t::tmp_storage_t dist;
        typename buffer_t::tmp_storage_t buffer;
    };

    tmp_storage_t* tmp_storage;
    // matrix of query vectors in the column-major layout in global memory
    const float* queries;
    // matrix of DB vectors in the column-major layout in global memory
    const float* db;
    // dimension of all vectors in `queries` and `db`
    std::int32_t dim;
    // total number of query vectors
    std::int32_t num_queries;
    // total number of DB vectors
    std::int32_t num_db;
    // output distance matrix
    float* out_dist;
    // output label matrix
    std::int32_t* out_label;

    /** Set shared memory used by this kernel.
     *
     * @param shm reference to the shared memory.
     */
    __device__ __forceinline__ void set_tmp_storage(tmp_storage_t* shm) { tmp_storage = shm; }

    /** Run the fused cache kernel.
     */
    __device__ __forceinline__ void run()
    {
        // distance submatrix computed by this thread
        float dist[QUERY_REG][DB_REG];
        // kth smallest distance found so far
        float radius[QUERY_REG];
        // portion of the top k lists stored in this thread
        float topk_dist[ITEMS_PER_THREAD];
        std::int32_t topk_label[ITEMS_PER_THREAD];
        // linear thread ID
        const std::int32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;

        // initialize radii to infinity
#pragma unroll
        for (std::int32_t i = 0; i < QUERY_REG; ++i)
        {
            radius[i] = std::numeric_limits<float>::infinity();
        }

        // initialize the top k list
#pragma unroll
        for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            topk_dist[i] = std::numeric_limits<float>::infinity();
        }

        // skip to the first query assigned to this thread block
        queries += blockIdx.x * QUERY_TILE;

        buffer_t buffer{tmp_storage->buffer};
        dist_t distance{tmp_storage->dist};

        __syncthreads(); // because of shared memory initialization in buffer

        std::int32_t num_valid_queries = num_queries - blockIdx.x * QUERY_TILE;
        if (num_valid_queries > QUERY_TILE)
        {
            num_valid_queries = QUERY_TILE;
        }
        std::int32_t db_offset = 0;
        for (; db_offset + DB_TILE <= num_db; db_offset += DB_TILE, db += DB_TILE)
        {
            distance.compute(queries, db, /*queries_stride=*/num_queries, /*db_stride=*/num_db, dim,
                             num_valid_queries, DB_TILE, dist);

            std::int32_t buffer_pos[QUERY_REG][DB_REG];
            buffer.alloc(dist, radius, buffer_pos, num_valid_queries, DB_TILE);
            buffer.insert(dist, buffer_pos, topk_dist, topk_label, radius, db_offset);
        }

        // the last iteration with bound checks
        if (db_offset < num_db)
        {
            const std::int32_t num_valid_db = num_db - db_offset;
            distance.compute(queries, db, /*queries_stride=*/num_queries, /*db_stride=*/num_db, dim,
                             num_valid_queries, num_valid_db, dist);

            // add the computed distances to the buffer
            std::int32_t buffer_pos[QUERY_REG][DB_REG];
            buffer.alloc(dist, radius, buffer_pos, num_valid_queries, num_valid_db);
            buffer.insert(dist, buffer_pos, topk_dist, topk_label, radius, db_offset);
        }

        // merge the remainder of values in the buffers
        buffer.merge(topk_dist, topk_label);

        __syncthreads();

        // store the results to the global memory
#pragma unroll
        for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            const auto idx = thread_idx * ITEMS_PER_THREAD + i;
            const auto array_idx = idx / K;
            const auto element_idx = idx % K;
            const auto query_idx = array_idx + blockIdx.x * QUERY_TILE;

            if (query_idx < num_queries && array_idx < QUERY_TILE)
            {
                out_dist[query_idx * K + element_idx] = topk_dist[i];
                out_label[query_idx * K + element_idx] = topk_label[i];
            }
        }
    }
};

#endif // BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_KERNEL_STRUCTS_CUH_
