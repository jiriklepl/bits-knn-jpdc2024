#include <cstddef>
#include <utility>

#include "bits/array_view.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"

#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/distance/cuda_distance.hpp"

void cuda_distance::prepare(const knn_args& args)
{
    args_ = args;

    dist_gpu_.release();
    points_gpu_.release();
    queries_gpu_.release();

    // allocate memory
    if (args_.dist_layout == matrix_layout::row_major)
    {
        dist_gpu_ = cuda_array<float, 2>{{args_.query_count, args_.point_count}};
    }
    else // column major
    {
        dist_gpu_ = cuda_array<float, 2>{{args_.point_count, args_.query_count}};
    }

    if (args_.points_layout == matrix_layout::row_major)
    {
        points_gpu_ = cuda_array<float, 2>{{args_.point_count, args_.dim}};
    }
    else // column major
    {
        points_gpu_ = cuda_array<float, 2>{{args_.dim, args_.point_count}};
    }

    if (args_.queries_layout == matrix_layout::row_major)
    {
        queries_gpu_ = cuda_array<float, 2>{{args_.query_count, args_.dim}};
    }
    else // column major
    {
        queries_gpu_ = cuda_array<float, 2>{{args_.dim, args_.query_count}};
    }

    // copy the input and initialize values
    cuda_stream::make_default()
        .copy_to_gpu_async(points_gpu_.view(), args_.points)
        .copy_to_gpu_async(queries_gpu_.view(), args_.queries)
        .fill_async<float>(dist_gpu_.view().data(), dist_gpu_.view().size(), 0)
        .sync();
}

array_view<float, 2> cuda_distance::matrix_gpu() const
{
    // return view with size `query_count * point_count` even if the allocated block of memory is a
    // larger matrix
    std::size_t size[2] = {args_.query_count, args_.point_count};

    if (args_.dist_layout == matrix_layout::column_major)
    {
        std::swap(size[0], size[1]);
    }

    auto dist = dist_gpu_.view();
    return array_view<float, 2>{dist.data(), size, {dist.stride(0), dist.stride(1)}};
}

array_view<float, 2> cuda_distance::matrix_cpu()
{
    auto dist_gpu = matrix_gpu();

    dist_cpu_.resize(dist_gpu.size(0) * dist_gpu.size(1));

    cuda_stream::make_default().copy_from_gpu_async(dist_cpu_.data(), dist_gpu).sync();

    return array_view<float, 2>{
        dist_cpu_.data(), {dist_gpu.size(0), dist_gpu.size(1)}, {dist_gpu.size(1), 1}};
}
