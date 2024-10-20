#include <cassert>
#include <cstddef>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/distance/cublas_distance.hpp"

namespace
{

/** Compute squared Euclidean distances of @p points
 *
 * @tparam LAYOUT layout of the @p points matrix
 * @param[in] points matrix of database vectors
 * @param[out] lengths computed @p points norms
 */
template <matrix_layout LAYOUT>
__global__ void squared_length_kernel(array_view<float, 2> points, float* lengths)
{
    constexpr std::size_t POINTS_DIM = LAYOUT == matrix_layout::row_major ? 0 : 1;
    constexpr std::size_t OTHER_DIM = 1 - POINTS_DIM;

    const auto points_count = points.size(POINTS_DIM);
    const auto dim = points.size(OTHER_DIM);
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= points_count)
    {
        return;
    }

    // compute the length
    float length = 0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (LAYOUT == matrix_layout::row_major)
        {
            length += points(idx, i) * points(idx, i);
        }
        else // column major
        {
            length += points(i, idx) * points(i, idx);
        }
    }

    // store the length to the output
    lengths[idx] = length;
}

/** Add database vector norms to the computed distance matrix.
 *
 * @tparam BLOCK_SIZE number of threads in each thread block
 * @tparam QUERIES_PER_BLOCK number of queries processed by each thread block
 * @tparam POINTS_PER_BLOCK number of database vectors processed by each thread block
 * @param[in] norms precomputed norms
 * @param[in,out] dist matrix with dot products. This kernel adds @p norms to this matrix
 */
template <std::size_t BLOCK_SIZE, std::size_t QUERIES_PER_BLOCK, std::size_t POINTS_PER_BLOCK>
__global__ void add_norm_kernel(const float* norms, array_view<float, 2> dist)
{
    constexpr std::size_t POINTS_PER_THREAD = POINTS_PER_BLOCK / BLOCK_SIZE;

    const auto query_count = dist.size(0);
    const auto point_count = dist.size(1);

    // load precomputed norms to registers of this thread block
    float norms_reg[POINTS_PER_THREAD];
#pragma unroll
    for (std::size_t i = 0; i < POINTS_PER_THREAD; ++i)
    {
        const auto point_idx = blockIdx.x * POINTS_PER_BLOCK + threadIdx.x + i * BLOCK_SIZE;
        norms_reg[i] = point_idx < point_count ? norms[point_idx] : 0.f;
    }

    // add the norms to the distance matrix
#pragma unroll
    for (std::size_t i = 0; i < QUERIES_PER_BLOCK; ++i)
    {
        const auto query_idx = blockIdx.y * QUERIES_PER_BLOCK + i;
#pragma unroll
        for (std::size_t j = 0; j < POINTS_PER_THREAD; ++j)
        {
            const auto point_idx = blockIdx.x * POINTS_PER_BLOCK + threadIdx.x + j * BLOCK_SIZE;
            if (query_idx < query_count && point_idx < point_count)
            {
                dist(query_idx, point_idx) += norms_reg[j];
            }
        }
    }
}

} // namespace

cublas_distance::cublas_distance() { cublasCreate(&handle_); }

cublas_distance::~cublas_distance() { cublasDestroy(handle_); }

void cublas_distance::prepare(const knn_args& args)
{
    cuda_distance::prepare(args);

    lengths_ = make_cuda_ptr<float>(args_.point_count);
}

void cublas_distance::compute()
{
    auto points = points_gpu_.view();
    auto queries = queries_gpu_.view();
    auto dist = dist_gpu_.view();

    // compute matrix: -2 * (dot product between each pair of points)
    // note, cuBLAS expects the matrices to be in a column major order
    const float alpha = -2;
    const float beta = 1;
    cublasStatus_t status = CUBLAS_STATUS_INVALID_VALUE;
    if (args_.points_layout == matrix_layout::row_major &&
        args_.queries_layout == matrix_layout::row_major)
    {
        status = cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, args_.point_count,
                             args_.query_count, args_.dim, &alpha, points.data(), points.stride(0),
                             queries.data(), queries.stride(0), &beta, dist.data(), dist.stride(0));
    }
    else if (args_.points_layout == matrix_layout::row_major &&
             args_.queries_layout == matrix_layout::column_major)
    {
        status = cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T, args_.point_count,
                             args_.query_count, args_.dim, &alpha, points.data(), points.stride(0),
                             queries.data(), queries.stride(0), &beta, dist.data(), dist.stride(0));
    }
    else if (args_.points_layout == matrix_layout::column_major &&
             args_.queries_layout == matrix_layout::row_major)
    {
        status = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, args_.point_count,
                             args_.query_count, args_.dim, &alpha, points.data(), points.stride(0),
                             queries.data(), queries.stride(0), &beta, dist.data(), dist.stride(0));
    }
    else if (args_.points_layout == matrix_layout::column_major &&
             args_.queries_layout == matrix_layout::column_major)
    {
        status = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, args_.point_count,
                             args_.query_count, args_.dim, &alpha, points.data(), points.stride(0),
                             queries.data(), queries.stride(0), &beta, dist.data(), dist.stride(0));
    }

    assert(status == CUBLAS_STATUS_SUCCESS);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error\n";
    }

    // compute the norm
    const auto block_size = args_.dist_block_size;
    const auto block_count = (args_.point_count + block_size - 1) / block_size;
    if (args_.points_layout == matrix_layout::row_major)
    {
        squared_length_kernel<matrix_layout::row_major>
            <<<block_count, block_size>>>(points, lengths_.get());
    }
    else
    {
        squared_length_kernel<matrix_layout::column_major>
            <<<block_count, block_size>>>(points, lengths_.get());
    }

    if (postprocessing_)
    {
        constexpr std::size_t BLOCK_SIZE = 128;
        constexpr std::size_t QUERIES_PER_BLOCK = 16;
        constexpr std::size_t POINTS_PER_BLOCK = BLOCK_SIZE * 4;
        const auto add_block_size = dim3(BLOCK_SIZE, 1, 1);
        const auto add_block_count =
            dim3((args_.point_count + POINTS_PER_BLOCK - 1) / POINTS_PER_BLOCK,
                 (args_.query_count + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK, 1);
        add_norm_kernel<BLOCK_SIZE, QUERIES_PER_BLOCK, POINTS_PER_BLOCK>
            <<<add_block_count, add_block_size>>>(lengths_.get(), dist);
    }
}
