#include <cstddef>
#include <utility>
#include <vector>

#include "bits/cuda_array.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/distance/magma_distance.hpp"

#include "bits/distance/abstract_gemm.cuh"

void magma_distance::prepare(const knn_args& args)
{
    args_ = args;

    // allocate matrices in GPU memory space
    points_gpu_ = cuda_array<float, 2>{{args_.dim, args_.point_count}};
    queries_gpu_ = cuda_array<float, 2>{{args_.dim, args_.query_count}};

    if (args_.dist_layout == matrix_layout::row_major)
    {
        dist_gpu_ =
            cuda_array<float, 2>{{args_.query_count, args_.point_count}, {args_.point_count, 1}};
    }
    else // column major
    {
        dist_gpu_ =
            cuda_array<float, 2>{{args_.point_count, args_.query_count}, {args_.query_count, 1}};
    }

    std::vector<float> points;
    std::vector<float> queries;

    // transpose the database matrix if necessary
    if (args_.points_layout == matrix_layout::row_major)
    {
        points.resize(args_.dim * args_.point_count);
        for (std::size_t i = 0; i < args_.point_count; ++i)
        {
            for (std::size_t j = 0; j < args_.dim; ++j)
            {
                points[j * args_.point_count + i] = args_.points[i * args_.dim + j];
            }
        }
        args_.points = points.data();
    }

    // transpose the query matrix if necessary
    if (args_.queries_layout == matrix_layout::row_major)
    {
        queries.resize(args_.dim * args_.query_count);
        for (std::size_t i = 0; i < args_.query_count; ++i)
        {
            for (std::size_t j = 0; j < args_.dim; ++j)
            {
                queries[j * args_.query_count + i] = args_.queries[i * args_.dim + j];
            }
        }
        args_.queries = queries.data();
    }

    // copy the aligned matrices to the GPU
    transfer_begin_.record();
    cuda_stream::make_default()
        .copy_to_gpu_async(points_gpu_.view().data(), args_.points, args_.dim * args_.point_count)
        .copy_to_gpu_async(queries_gpu_.view().data(), args_.queries, args_.dim * args_.query_count)
        .sync();
    transfer_end_.record();
    transfer_end_.sync();
}

void magma_distance::compute()
{
    auto dist = dist_gpu_.view();
    auto db = points_gpu_.view();
    auto queries = queries_gpu_.view();

    if (args_.dist_layout == matrix_layout::column_major)
    {
        std::swap(db, queries);
    }

    run_abstract_gemm<l2_ops>(db.size(0), queries.size(1), db.size(1), queries.data(), db.data(),
                              dist.data());
}

void magma_partial_distance::compute()
{
    auto dist = dist_gpu_.view();
    auto db = points_gpu_.view();
    auto queries = queries_gpu_.view();

    // this distance function is not symmetric
    if (args_.dist_layout == matrix_layout::column_major)
    {
        throw std::runtime_error{"Partial MAGMA distance does not support the column-major layout "
                                 "of the distance matrix."};
    }

    run_abstract_gemm<partial_l2_ops>(db.size(0), queries.size(1), db.size(1), queries.data(),
                                      db.data(), dist.data());
}

void magma_kl_distance::compute()
{
    auto dist = dist_gpu_.view();
    auto db = points_gpu_.view();
    auto queries = queries_gpu_.view();

    if (args_.dist_layout == matrix_layout::column_major)
    {
        std::swap(db, queries);
    }

    run_abstract_gemm<kl_divergence_ops>(db.size(0), queries.size(1), db.size(1), queries.data(),
                                         db.data(), dist.data());
}
