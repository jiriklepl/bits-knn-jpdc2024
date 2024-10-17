#include <cstdint>
#include <stdexcept>

#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/knn.hpp"
#include "bits/topk/singlepass/fused_tc_kernel_runner.hpp"
#include "bits/topk/singlepass/fused_tc_knn.hpp"
#include "bits/topk/singlepass/fused_tc_policy.hpp"

#include "bits/topk/bitonic_sort_regs.cuh"

namespace
{

/** Auxiliary function to transform runtime variable values to template constants.
 */
template <typename Policy, std::int32_t K>
void run(fused_tc_kernel_runner<Policy>& kernel)
{
    if (!dynamic_switch<128, 256, 512>(kernel.block_size, [&]<std::size_t BlockSize>() {
        kernel.template operator()<K, BlockSize>();
    })) {
        throw std::runtime_error{"Unsupported block size: " + std::to_string(kernel.block_size)};
    }
}

/** Auxiliary function to transform runtime variable values to template constants.
 */
template <typename Policy>
void run(fused_tc_kernel_runner<Policy>& kernel)
{
    if (!dynamic_switch<2, 4, 8, 16, 32, 64, 128>(kernel.k, [&]<std::size_t K>() {
        run<Policy, K>(kernel);
    })) {
        throw std::runtime_error{"Unsupported k value: " + std::to_string(kernel.k)};
    }
}

} // namespace

template <typename Policy>
void fused_tc_knn<Policy>::initialize(const knn_args& args)
{
    // skip allocation in cuda_knn::initialize()
    knn::initialize(args);

    out_dist_gpu_ = cuda_array<float, 2>{{query_count(), k()}};
    out_label_gpu_ = cuda_array<std::int32_t, 2>{{query_count(), k()}};
    points_gpu_ = cuda_array<float, 2>{{point_count(), dim()}};
    queries_gpu_ = cuda_array<float, 2>{{query_count(), dim()}};

    const auto points = points_gpu_.view();
    const auto queries = queries_gpu_.view();

    cuda_stream::make_default()
        .copy_to_gpu_async(points, args_.points)
        .copy_to_gpu_async(queries, args_.queries)
        .sync();

    const auto aligned_dim = (points.size(1) + DIM_TILE_SIZE - 1) / DIM_TILE_SIZE * DIM_TILE_SIZE;
    const auto point_tiles = (points.size(0) * aligned_dim + DIM_TILE_SIZE * POINT_TILE_SIZE - 1) /
                             (DIM_TILE_SIZE * POINT_TILE_SIZE);
    in_points_gpu_ = cuda_array<input_t, 2>({point_tiles, DIM_TILE_SIZE * POINT_TILE_SIZE});
    in_point_norms_gpu_ = cuda_array<float, 1>({points.size(0)});

    assert(aligned_dim == (queries.size(1) + DIM_TILE_SIZE - 1) / DIM_TILE_SIZE * DIM_TILE_SIZE);
    const auto query_tiles = (queries.size(0) * aligned_dim + DIM_TILE_SIZE * QUERY_TILE_SIZE - 1) /
                             (DIM_TILE_SIZE * QUERY_TILE_SIZE);
    in_queries_gpu_ = cuda_array<input_t, 2>({query_tiles, DIM_TILE_SIZE * QUERY_TILE_SIZE});
    in_query_norms_gpu_ = cuda_array<float, 1>({queries.size(0)});
}

template <typename Policy>
void fused_tc_knn<Policy>::selection()
{
    auto points = points_gpu_.view();
    auto queries = queries_gpu_.view();

    fused_tc_kernel_runner<Policy> kernel;
    kernel.points = points;
    kernel.queries = queries;

    kernel.in_points = in_points_gpu_.view();
    kernel.in_point_norms = in_point_norms_gpu_.view();
    kernel.in_queries = in_queries_gpu_.view();
    kernel.in_query_norms = in_query_norms_gpu_.view();

    kernel.out_dist = out_dist_gpu();
    kernel.out_label = out_label_gpu();

    kernel.k = k();
    kernel.block_size = selection_block_size();

    run(kernel);

    cuda_stream::make_default().sync();
}

template void fused_tc_knn<fused_tc_half_policy>::initialize(const knn_args& args);
template void fused_tc_knn<fused_tc_half_policy>::selection();

template void fused_tc_knn<fused_tc_bfloat16_policy>::initialize(const knn_args& args);
template void fused_tc_knn<fused_tc_bfloat16_policy>::selection();

template void fused_tc_knn<fused_tc_double_policy>::initialize(const knn_args& args);
template void fused_tc_knn<fused_tc_double_policy>::selection();
