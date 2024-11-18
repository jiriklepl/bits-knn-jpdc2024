#ifndef BITS_DISTANCE_ABSTRACT_GEMM_CUH_
#define BITS_DISTANCE_ABSTRACT_GEMM_CUH_

#include <cassert>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "bits/cuch.hpp"
#include "bits/cuda_stream.hpp"

/** Operators for dot products in matrix multiplication.
 */
struct dot_product_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 6;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 6;

    /** Compute dot product and add it to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N]) const
    {
        // compute the distances
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                acc[i][j] += lhs[i] * rhs[j];
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&)[REGS_M][REGS_N]) const {}
};

/** Operators for squared L2 distances in matrix multiplication.
 */
struct l2_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 6;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 6;

    /** Compute squared L2 distances and add them to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N]) const
    {
        // compute the distances
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                const float diff = lhs[i] - rhs[j];
                acc[i][j] = fmaf(diff, diff, acc[i][j]);
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&)[REGS_M][REGS_N]) const {}
};

/** Operators for partial squared L2 distances in matrix multiplication.
 *
 * This computes `norm(x)^2 - 2 * dot(x, y)` where `x` is a DB vector and `y` is a query vector.
 * Note that the computed quantity is almost a squared Euclidean distance (it suffices to add
 * `norm(y)^2` to make it a squared Euclidean distance).
 */
struct partial_l2_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 6;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 6;

    float db_norm[REGS_N];

    __device__ __forceinline__ partial_l2_ops()
    {
#pragma unroll
        for (int j = 0; j < REGS_N; ++j)
        {
            db_norm[j] = 0;
        }
    }

    /** Compute squared L2 distances and add them to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N])
    {
#pragma unroll
        for (int j = 0; j < REGS_N; ++j)
        {
            db_norm[j] += rhs[j] * rhs[j];
#pragma unroll
            for (int i = 0; i < REGS_M; ++i)
            {
                acc[i][j] += lhs[i] * rhs[j];
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&acc)[REGS_M][REGS_N]) const
    {
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                acc[i][j] = db_norm[j] - 2 * acc[i][j];
            }
        }
    }
};

/** Operators for computing KL divergence.
 *
 * It assumes that the first input matrix contrains sample distributions and the second input
 * matrix contrains baseline distributions.
 */
struct kl_divergence_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 4;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 4;

    /** Compute KL divergence and add it to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N]) const
    {
        // compute log(rhs)
        float log_rhs[REGS_N];
#pragma unroll
        for (int j = 0; j < REGS_N; ++j)
        {
            log_rhs[j] = __logf(rhs[j]);
        }

#pragma unroll
        for (int i = 0; i < REGS_N; ++i)
        {
            lhs[i] = __logf(lhs[i]);
        }

        // compute the distances
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                acc[i][j] += rhs[j] * (log_rhs[j] - lhs[i]);
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&)[REGS_M][REGS_N]) const {}
};

/** Operators for computing LP distance.
 */
template <int POW>
struct lp_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 6;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 6;

    /** Compute squared L2 distances and add them to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N]) const
    {
// compute the distances
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                const float diff = abs(lhs[i] - rhs[j]);
                float res = 1;
#pragma unroll
                for (int exp = 0; exp < POW; ++exp)
                {
                    res *= diff;
                }
                acc[i][j] += res;
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&)[REGS_M][REGS_N]) const {}
};

/** Operators for lower bound calculations in pivot-based methods.
 */
struct lower_bound_ops
{
    // dimension of thread blocks along the M dimension
    static constexpr int BLOCK_DIM_M = 16;
    // dimension of thread blocks along the N dimension
    static constexpr int BLOCK_DIM_N = 16;
    // number of registers per thread along the M dimension
    static constexpr int REGS_M = 6;
    // number of registers per thread along the N dimension
    static constexpr int REGS_N = 6;

    /** Compute dot product and add it to the matrix @p acc
     *
     * @p lhs and @p rhs contain vector components of input vectors along the same dimension.
     * Combination of all pairs of @p lhs and @p rhs are added to @p acc
     *
     * @param[in] lhs values of `REGS_M` vector components from the first matrix.
     * @param[in] rhs values of `REGS_N` vector components from the second matrix.
     * @param[in,out] acc result matrix accumulator.
     */
    __device__ __forceinline__ void run(float (&lhs)[REGS_M], float (&rhs)[REGS_N],
                                        float (&acc)[REGS_M][REGS_N]) const
    {
        // compute the distances
#pragma unroll
        for (int i = 0; i < REGS_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                acc[i][j] = fmaxf(acc[i][j], abs(lhs[i] - rhs[j]));
            }
        }
    }

    /** Finish the computation.
     *
     * @param acc computed distance
     */
    __device__ __forceinline__ void finish(float (&)[REGS_M][REGS_N]) const {}
};

/** Matrix pointer wrapper which checks if there are out-of-bounds accesses.
 *
 * If a column or a row is accessed outside of the size of the matrix, the dereference operator
 * returns 0 instead of loading the data from memory.
 *
 * @tparam T type of values
 */
template <typename T>
struct guarded_ptr
{
    T* ptr;
    std::int64_t row;
    std::int64_t col;
    std::int64_t num_rows;
    std::int64_t num_cols;
    std::int64_t stride;
    T dummy_ = static_cast<T>(0);

    /** Create a guarded pointer
     *
     * @param ptr pointer to the matrix
     * @param row current row @p ptr points to
     * @param col current column @p ptr points to
     * @param num_rows total number of rows in the matrix
     * @param num_cols total number of columns in the matrix
     */
    __device__ __forceinline__ guarded_ptr(T* ptr, std::int64_t row, std::int64_t col,
                                           std::int64_t num_rows, std::int64_t num_cols,
                                           std::int64_t stride)
        : ptr(ptr), row(row), col(col), num_rows(num_rows), num_cols(num_cols), stride(stride)
    {
    }

    __device__ __forceinline__ T operator[](int col_offset) const
    {
        return row < num_rows && col + col_offset < num_cols ? ptr[col_offset] : identity();
    }

    __device__ __forceinline__ T& operator[](int col_offset)
    {
        return row < num_rows && col + col_offset < num_cols ? ptr[col_offset] : dummy_;
    }

    __device__ __forceinline__ void next_row(std::int64_t row_offset)
    {
        ptr += row_offset * stride;
        row += row_offset;
    }

    __device__ __forceinline__ guarded_ptr& operator+=(std::int64_t offset)
    {
        // we assume offset is a multiple of row size
        assert(offset % stride == 0);

        ptr += offset;
        row += offset / stride;
        return *this;
    }

    __device__ __forceinline__ T identity() const { return static_cast<T>(0); }
};

/** Pointer wrapper that does not check if there are out-of-bounds accesses.
 *
 * @tparam T type of the data
 */
template <typename T>
struct unguarded_ptr
{
    T* ptr;
    std::int64_t stride;

    /** Create a plain pointer wrapper
     *
     * @param ptr pointer to the matrix
     * @param row unused (to conform to the guarded pointer interface)
     * @param col unused (to conform to the guarded pointer interface)
     * @param num_rows unused (to conform to the guarded pointer interface)
     * @param num_cols unused (to conform to the guarded pointer interface)
     */
    __device__ __forceinline__ unguarded_ptr(T* ptr, std::int64_t, std::int64_t, std::int64_t,
                                             std::int64_t, std::int64_t stride)
        : ptr(ptr), stride(stride)
    {
    }

    __device__ __forceinline__ void next_row(std::int64_t row_offset)
    {
        ptr += stride * row_offset;
    }

    __device__ __forceinline__ T operator[](int col_offset) const { return ptr[col_offset]; }

    __device__ __forceinline__ T& operator[](int col_offset) { return ptr[col_offset]; }

    __device__ __forceinline__ unguarded_ptr& operator+=(std::int64_t offset)
    {
        ptr += offset;
        return *this;
    }
};

/** A modification of the kernel due to Li et al. from https://github.com/geomlab-ucd/bf-knn
 *
 * Tiles from input matrices are cached in shared memory. The kernel also uses registers to cache
 * parts of the tiles from shared memory in registers.
 *
 * @tparam Operators abstract multiplication/addition operators
 * @tparam BLOCK_DIM_M number of threads per thread block along the M dimension
 * @tparam BLOCK_DIM_N number of threads per thread block along the N dimension
 * @tparam OUT_REGS_M number of registers along the M dimension used per thread to hold a tile of
 * the output matrix
 * @tparam OUT_REGS_N number of registers along the N dimension used per thread to hold a tile of
 * the output matrix
 *
 * @param K dimension of all input vectors in input matrices (number of rows of input matrices)
 * @param M number of columns in the first matrix @p A
 * @param N number of columns in the second matrix @p B
 * @param stride_a stride to get to the next row of matrix @p A
 * @param stride_b stride to get to the next row of matrix @p B
 * @param A the first input matrix -- @p K by @p M matrix
 * @param B the second input matrix -- @p K by @p N matrix
 * @param C memory for the output matrix -- @p M by @p N matrix (`C = A' * B` where the prime '
 * denotes matrix transposition)
 */
/*******************************************************************************
 * bf-knn (Brute-Force k-Nearest Neighbors Search on the GPU) is the proprietary
 * property of The Regents of the University of California ("The Regents.")
 *
 * Copyright © 2015 The Regents of the University of California, Davis campus.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted by nonprofit, research institutions for research
 * use only, provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of The Regents may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * The end-user understands that the program was developed for research purposes
 * and is advised not to rely exclusively on the program for any reason.
 *
 * THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO
 * OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. THE REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL THE REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
 * INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR
 * BUSINESS INTERRUPTION, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY
 * WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * If you do not agree to these terms, do not download or use the software. This
 * license may be modified only in a writing signed by authorized signatory of
 * both parties.
 *
 * For commercial license information please contact copyright@ucdavis.edu.
 ******************************************************************************/
template <class Operators, int BLOCK_DIM_M, int BLOCK_DIM_N, int OUT_REGS_M, int OUT_REGS_N>
__global__ void __launch_bounds__(BLOCK_DIM_M* BLOCK_DIM_N, 2)
    abstract_gemm(const int K, const int M, const int N, int stride_a, int stride_b, const float* A,
                  const float* B, float* C)
{
    // size of the tile along the M dimension
    constexpr int TILE_M = BLOCK_DIM_M * OUT_REGS_M;
    // size of the tile along the N dimension
    constexpr int TILE_N = BLOCK_DIM_N * OUT_REGS_N;
    // number of thread in a thread block
    constexpr int BLOCK_SIZE = BLOCK_DIM_N * BLOCK_DIM_M;
    // number of threads in a warp
    constexpr int WARP_SIZE = 32;
    // number of registers per thread along the M dimension
    constexpr int REGS_M = (TILE_M + WARP_SIZE - 1) / WARP_SIZE;
    // number of registers per thread along the N dimension
    constexpr int REGS_N = (TILE_N + WARP_SIZE - 1) / WARP_SIZE;
    // number of registers per thread along the K dimension (K is the vector dimension in A and B)
    constexpr int REGS_K = 2;
    // number of warps in a thread block
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // number of items along the K dimension processed per tile
    constexpr int TILE_K = REGS_K * NUM_WARPS;

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * BLOCK_DIM_N + tx;
    const int tx2 = tid % WARP_SIZE;
    const int ty2 = tid / WARP_SIZE;

    volatile __shared__ float as[TILE_K][TILE_M];
    volatile __shared__ float bs[TILE_K][TILE_N];

    float cr[OUT_REGS_M][OUT_REGS_N];
    float ar[OUT_REGS_M];
    float br[OUT_REGS_N];

    float asr[REGS_K][REGS_M];
    float bsr[REGS_K][REGS_N];

    A += static_cast<std::int64_t>(ty2) * stride_a + (by * TILE_M + tx2);
    B += static_cast<std::int64_t>(ty2) * stride_b + (bx * TILE_N + tx2);
    C += static_cast<std::int64_t>(by * TILE_M + ty) * N + (bx * TILE_N + tx);

    Operators ops;
    guarded_ptr global_a{A, ty2, by * TILE_M + tx2, K, M, stride_a};
    guarded_ptr global_b{B, ty2, bx * TILE_N + tx2, K, N, stride_b};
    guarded_ptr global_c{C, by * TILE_M + ty, bx * TILE_N + tx, M, N, N};

// Zero C reg
#pragma unroll
    for (int i = 0; i < OUT_REGS_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < OUT_REGS_N; ++j)
        {
            cr[i][j] = 0.0F;
        }
    }

// Load A gmem->smem
#pragma unroll
    for (int i = 0; i < REGS_K; ++i)
    {
#pragma unroll
        for (int j = 0; j < REGS_M; ++j)
        {
            if (TILE_M % WARP_SIZE == 0 || j * WARP_SIZE + tx2 < TILE_M)
            {
                as[i * NUM_WARPS + ty2][j * WARP_SIZE + tx2] =
                    global_a[j * WARP_SIZE]; // A[j * 32];
            }
        }
        // A += M * 8;
        global_a.next_row(static_cast<std::int64_t>(NUM_WARPS));
    }

// Load B gmem->smem
#pragma unroll
    for (int i = 0; i < REGS_K; ++i)
    {
#pragma unroll
        for (int j = 0; j < REGS_N; ++j)
        {
            if (TILE_N % WARP_SIZE == 0 || j * WARP_SIZE + tx2 < TILE_N)
            {
                bs[i * NUM_WARPS + ty2][j * WARP_SIZE + tx2] =
                    global_b[j * WARP_SIZE]; // B[j * 32];
            }
        }
        // B += N * 8;
        global_b.next_row(static_cast<std::int64_t>(NUM_WARPS));
    }

    __syncthreads();

    for (int kk = 0; kk < K - TILE_K; kk += TILE_K)
    {
// Load A gmen->reg
#pragma unroll
        for (int i = 0; i < REGS_K; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_M; ++j)
            {
                asr[i][j] = global_a[j * WARP_SIZE]; // A[j * 32];
            }
            // A += M * 8;
            global_a.next_row(static_cast<std::int64_t>(NUM_WARPS));
        }

// Load B gmem->reg
#pragma unroll
        for (int i = 0; i < REGS_K; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                bsr[i][j] = global_b[j * WARP_SIZE]; // B[j * 32];
            }
            // B += N * 8;
            global_b.next_row(static_cast<std::int64_t>(NUM_WARPS));
        }

// Compute
#pragma unroll
        for (int k = 0; k < TILE_K; ++k)
        {
// Load B smen->reg
#pragma unroll
            for (int j = 0; j < OUT_REGS_N; ++j)
            {
                br[j] = bs[k][j * BLOCK_DIM_N + tx];
            }
// Load A smen->reg
#pragma unroll
            for (int i = 0; i < OUT_REGS_M; ++i)
            {
                ar[i] = as[k][i * BLOCK_DIM_M + ty];
            }

            // compute
            ops.run(ar, br, cr);
        }

        __syncthreads();

// Load A reg->smem
#pragma unroll
        for (int i = 0; i < REGS_K; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_M; ++j)
            {
                if (TILE_M % WARP_SIZE == 0 || j * WARP_SIZE + tx2 < TILE_M)
                {
                    as[i * NUM_WARPS + ty2][j * WARP_SIZE + tx2] = asr[i][j];
                }
            }
        }

// Load B reg->smem
#pragma unroll
        for (int i = 0; i < REGS_K; ++i)
        {
#pragma unroll
            for (int j = 0; j < REGS_N; ++j)
            {
                if (TILE_N % WARP_SIZE == 0 || j * WARP_SIZE + tx2 < TILE_N)
                {
                    bs[i * NUM_WARPS + ty2][j * WARP_SIZE + tx2] = bsr[i][j];
                }
            }
        }

        __syncthreads();
    }

// Compute last 16 dimensions
#pragma unroll
    for (int k = 0; k < TILE_K; ++k)
    {
// Load B smen->reg
#pragma unroll
        for (int j = 0; j < OUT_REGS_N; ++j)
        {
            br[j] = bs[k][j * BLOCK_DIM_N + tx];
        }
// Load A smen->reg
#pragma unroll
        for (int i = 0; i < OUT_REGS_M; ++i)
        {
            ar[i] = as[k][i * BLOCK_DIM_M + ty];
        }

        // compute
        ops.run(ar, br, cr);
    }

    ops.finish(cr);

// Store C reg->gmem
#pragma unroll
    for (int i = 0; i < OUT_REGS_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < OUT_REGS_N; ++j)
        {
            // long long c = (long long)__float_as_int(cr[i][j]);
            // c = (c << 32) | (bx * 96 + j * 16 + tx);
            // C[j * 16] = ops.finish(cr[i][j]); // c;
            global_c[j * BLOCK_DIM_N] = cr[i][j];
        }
        // C += N * 16;
        global_c.next_row(static_cast<std::int64_t>(BLOCK_DIM_M));
    }
}

/** General matrix multiplication using abstract operators for multiplication and addition.
 *
 * @tparam Operators abstract multiplicative/additive operators used in matrix multiplication.
 * @param K number of rows in both matrices (i.e., @p A and @p B )
 * @param M number of columns in matrix @p A
 * @param N number of columns in matrix @p B
 * @param stride_a stride to get to the next row of matrix @p A
 * @param stride_b stride to get to the next row of matrix @p B
 * @param[in] A float matrix of size @p K * @p M (first operand)
 * @param[in] B float matrix of size @p K * @p N (second operand)
 * @param[out] C float matrix of size @p M * @p N (output matrix): `C = A' * B` where ' denotes
 *               matrix transposition.
 * @param stream cuda stream
 */
template <class Operators>
inline void run_abstract_gemm(const int K, const int M, const int N, int stride_a, int stride_b,
                              const float* A, const float* B, float* C, cuda_stream& stream)
{
    constexpr int BLOCK_DIM_M = Operators::BLOCK_DIM_M;
    constexpr int BLOCK_DIM_N = Operators::BLOCK_DIM_N;
    constexpr int OUT_REGS_M = Operators::REGS_M;
    constexpr int OUT_REGS_N = Operators::REGS_N;
    constexpr int TILE_SIZE_M = BLOCK_DIM_M * OUT_REGS_M;
    constexpr int TILE_SIZE_N = BLOCK_DIM_N * OUT_REGS_N;

    dim3 block(BLOCK_DIM_N, BLOCK_DIM_M);
    dim3 grid((N + TILE_SIZE_N - 1) / TILE_SIZE_N, (M + TILE_SIZE_M - 1) / TILE_SIZE_M);

    abstract_gemm<Operators, BLOCK_DIM_M, BLOCK_DIM_N, OUT_REGS_M, OUT_REGS_N>
        <<<grid, block, 0, stream.get()>>>(K, M, N, stride_a, stride_b, A, B, C);
    CUCH(cudaGetLastError());
}

template <class Operators>
inline void run_abstract_gemm(const int K, const int M, const int N, const float* A, const float* B,
                              float* C)
{
    auto stream = cuda_stream::make_default();
    run_abstract_gemm<Operators>(K, M, N, /*stride_a=*/M, /*stride_b=*/N, A, B, C, stream);
}

#endif // BITS_DISTANCE_ABSTRACT_GEMM_CUH_
