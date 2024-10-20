#include <cstddef>
#include <cstdint>

// #include <raft/core/device_resources.hpp>
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/detail/distance_ops/l2_exp.cuh>
#include <raft/distance/detail/distance_ops/l2_unexp.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn-inl.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/rapidsai_fused.hpp"

void rapidsai_fused::initialize(const knn_args& args)
{
    // skip allocation in cuda_knn::initialize()
    knn::initialize(args);

    out_dist_gpu_ = cuda_array<float, 2>{{query_count(), k()}};
    out_label_gpu_ = cuda_array<std::int32_t, 2>{{query_count(), k()}};

    if (row_major_query_)
    {
        queries_gpu_ = cuda_array<float, 2>{{query_count(), dim()}};

        cuda_stream::make_default().copy_to_gpu_async(queries_gpu_.view(), args_.queries);
    }
    else
    {
        queries_gpu_ = cuda_array<float, 2>{{dim(), query_count()}};
        std::vector<float> queries_transposed(queries_gpu_.view().size());

        for (std::size_t i = 0; i < query_count(); ++i)
        {
            for (std::size_t j = 0; j < dim(); ++j)
            {
                queries_transposed[j * query_count() + i] = args_.queries[i * dim() + j];
            }
        }

        cuda_stream::make_default().copy_to_gpu_async(queries_gpu_.view(),
                                                      queries_transposed.data());
    }

    if (row_major_index_)
    {
        points_gpu_ = cuda_array<float, 2>{{point_count(), dim()}};

        cuda_stream::make_default().copy_to_gpu_async(points_gpu_.view(), args_.points);
    }
    else
    {
        points_gpu_ = cuda_array<float, 2>{{dim(), point_count()}};
        std::vector<float> points_transposed(points_gpu_.view().size());

        for (std::size_t i = 0; i < point_count(); ++i)
        {
            for (std::size_t j = 0; j < dim(); ++j)
            {
                points_transposed[j * point_count() + i] = args_.points[i * dim() + j];
            }
        }

        cuda_stream::make_default().copy_to_gpu_async(points_gpu_.view(), points_transposed.data());
    }

    cuda_stream::make_default().sync();
}

void rapidsai_fused::distances()
{
    // no computation
}

void rapidsai_fused::selection()
{
    cuda_knn::selection();

    auto points = points_gpu_.view();
    auto queries = queries_gpu_.view();

    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    auto stream = cuda_stream::make_default();

    const auto metric = raft::distance::DistanceType::L2Expanded;

    raft::spatial::knn::detail::fusedL2Knn<std::int32_t, float>(
        dim(), out_label.data(), out_dist.data(), points.data(), queries.data(), point_count(),
        query_count(), k(), row_major_index_, row_major_query_, stream.get(), metric);

    cuda_stream::make_default().sync();
}
