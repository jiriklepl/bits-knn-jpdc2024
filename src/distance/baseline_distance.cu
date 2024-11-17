#include <cstddef>
#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/distance/baseline_distance.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/memory.cuh"

namespace
{

/** One thread per distance.
 *
 * @param A column major matrix
 * @param B column major matrix
 * @param distances @p A * @p B
 */
__global__ void distance_kernel(array_view<float, 2> A, array_view<float, 2> B,
                                array_view<float, 2> distances)
{
    const auto a_count = A.size(1);
    const auto b_count = B.size(1);
    const auto dim = A.size(0);

    // find index of a point and query
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto b_idx = idx / a_count;
    const auto a_idx = idx % a_count;

    if (b_idx >= b_count)
    {
        return;
    }

    // compute the distance
    float dist = 0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        const auto diff = B(i, b_idx) - A(i, a_idx);
        dist = fma(diff, diff, dist);
    }

    // write the result to the distance matrix
    distances(b_idx, a_idx) = dist;
}

} // namespace

void baseline_distance::prepare(const knn_args& args)
{
    args_ = args;

    dist_gpu_.release();
    points_gpu_.release();
    queries_gpu_.release();

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

    points_gpu_ = cuda_array<float, 2>{{args_.dim, args_.point_count}, {args_.point_count, 1}};
    queries_gpu_ = cuda_array<float, 2>{{args_.dim, args_.query_count}, {args_.query_count, 1}};

    // transpose points and queries to a column major layout (dimension * number of vectors)
    if (args_.points_layout == matrix_layout::row_major)
    {
        std::vector<float> data(args_.dim * args_.point_count);
        for (std::size_t i = 0; i < args_.point_count; ++i)
        {
            for (std::size_t j = 0; j < args_.dim; ++j)
            {
                data[j * args_.point_count + i] = args_.points[i * args_.dim + j];
            }
        }
        cuda_stream::make_default().copy_to_gpu_async(points_gpu_.view().data(), data.data(),
                                                      data.size());
    }

    if (args_.queries_layout == matrix_layout::row_major)
    {
        std::vector<float> data(args_.dim * args_.query_count);
        for (std::size_t i = 0; i < args_.query_count; ++i)
        {
            for (std::size_t j = 0; j < args_.dim; ++j)
            {
                data[j * args_.query_count + i] = args_.queries[i * args_.dim + j];
            }
        }
        cuda_stream::make_default().copy_to_gpu_async(queries_gpu_.view().data(), data.data(),
                                                      data.size());
    }

    cuda_stream::make_default()
        .fill_async<float>(dist_gpu_.view().data(), dist_gpu_.view().size(), 0)
        .sync();
}

void baseline_distance::compute()
{
    const auto block_size = args_.dist_block_size;
    const auto block_count = (args_.point_count * args_.query_count + block_size - 1) / block_size;

    auto a = points_gpu_.view();
    auto b = queries_gpu_.view();
    if (args_.dist_layout == matrix_layout::column_major)
    {
        swap_values(a, b);
    }
    distance_kernel<<<block_count, block_size>>>(a, b, dist_gpu_.view());
    CUCH(cudaGetLastError());
}
