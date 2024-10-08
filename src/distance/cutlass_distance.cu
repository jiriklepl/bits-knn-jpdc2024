#include <iostream>

#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>

#include "bits/array_view.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/distance/cutlass_distance.hpp"

// tag indicating euclidian distance computation
struct op_euclidean_distance;

// MMA specialization of cutlass to compute euclidean distances instead. We replace MAD with
// euclidian distance by providing a more concrete specialization of cutlass::arch::Mma (we
// explicitly set LayoutC). This is an ugly hack to compel CUTLASS to use our custom operator. It
// seems the proper way to do this would be to specialize several large configuration templates.
template <typename LayoutA, typename LayoutB>
struct cutlass::arch::Mma<cutlass::gemm::GemmShape<1, 1, 1>, 1, float, LayoutA, float, LayoutB,
                          float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>
{
    using Shape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;

    __forceinline__ __host__ __device__ void operator()(cutlass::Array<float, 1>& d,
                                                        const cutlass::Array<float, 1>& a,
                                                        const cutlass::Array<float, 1>& b,
                                                        const cutlass::Array<float, 1>& c)
    {
        const auto diff = a[0] - b[0];
        d[0] = diff * diff + c[0];
    }
};

// configuration of cutlass gemm for different layouts
template <matrix_layout LayoutPoints, matrix_layout LayoutQueries>
struct cutlass_configuration;

template <>
struct cutlass_configuration<matrix_layout::row_major, matrix_layout::row_major>
{
    using layout_queries = cutlass::layout::RowMajor;
    // transpose points by using column major layout
    using layout_points = cutlass::layout::ColumnMajor;
    // leading dimension index in query matrix
    static constexpr int ld_idx_query = 0;
    // leading dimension index in point matrix
    static constexpr int ld_idx_point = 0;
};

template <>
struct cutlass_configuration<matrix_layout::row_major, matrix_layout::column_major>
{
    using layout_queries = cutlass::layout::RowMajor;
    using layout_points = cutlass::layout::RowMajor;
    // leading dimension index in query matrix
    static constexpr int ld_idx_query = 0;
    // leading dimension index in point matrix
    static constexpr int ld_idx_point = 1;
};

template <>
struct cutlass_configuration<matrix_layout::column_major, matrix_layout::row_major>
{
    using layout_queries = cutlass::layout::ColumnMajor;
    using layout_points = cutlass::layout::ColumnMajor;
    // leading dimension index in query matrix
    static constexpr int ld_idx_query = 1;
    // leading dimension index in point matrix
    static constexpr int ld_idx_point = 0;
};

template <>
struct cutlass_configuration<matrix_layout::column_major, matrix_layout::column_major>
{
    using layout_queries = cutlass::layout::ColumnMajor;
    using layout_points = cutlass::layout::RowMajor;
    // leading dimension index in query matrix
    static constexpr int ld_idx_query = 1;
    // leading dimension index in point matrix
    static constexpr int ld_idx_point = 1;
};

namespace
{

template <matrix_layout LayoutPoints, matrix_layout LayoutQueries>
void call_cutlass(array_view<float, 2> queries, array_view<float, 2> points,
                  array_view<float, 2> dist)
{
    using config_t = cutlass_configuration<LayoutPoints, LayoutQueries>;

    using element_accumulator = float;
    using element_compute_epilogue = element_accumulator;
    using element_input_a = float;
    using element_input_b = float;
    using element_output = float;

    using layout_input_a = typename config_t::layout_queries;
    using layout_input_b = typename config_t::layout_points;
    using layout_output = cutlass::layout::RowMajor;

    using thread_block_shape = cutlass::gemm::GemmShape<128, 128, 8>;
    using warp_shape = cutlass::gemm::GemmShape<32, 32, 8>;
    using op_shape = cutlass::gemm::GemmShape<1, 1, 1>;

    using thread_block_swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using arch = cutlass::arch::Sm60;

    using epilogue_op =
        cutlass::epilogue::thread::LinearCombination<element_output, 1, element_accumulator,
                                                     element_compute_epilogue>;

    constexpr int stage_count = 2;

    using gemm_t = cutlass::gemm::device::Gemm<
        element_input_a, layout_input_a, element_input_b, layout_input_b, element_output,
        layout_output, element_accumulator, cutlass::arch::OpClassSimt, arch, thread_block_shape,
        warp_shape, op_shape, epilogue_op, thread_block_swizzle, stage_count, 1, 1, false,
        op_euclidean_distance>;

    const int m = static_cast<int>(queries.size(config_t::ld_idx_query));
    const int n = static_cast<int>(points.size(config_t::ld_idx_point));
    const int k = static_cast<int>(queries.size(1 - config_t::ld_idx_query));

    const typename gemm_t::Arguments gemm_args{
        {m, n, k},
        {queries.data(), static_cast<int>(queries.stride(0))},
        {points.data(), static_cast<int>(points.stride(0))},
        {dist.data(), static_cast<int>(dist.stride(0))},
        {dist.data(), static_cast<int>(dist.stride(0))},
        {1, 0},
        1};

    gemm_t op;
    const auto status = op(gemm_args);
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS error: " << cutlassGetStatusString(status) << std::endl;
    }
}

} // namespace

void cutlass_distance::prepare(const knn_args& args) { cuda_distance::prepare(args); }

void cutlass_distance::compute()
{
    auto queries = queries_gpu_.view();
    auto points = points_gpu_.view();
    auto dist = dist_gpu_.view();

    if (args_.queries_layout == matrix_layout::row_major &&
        args_.points_layout == matrix_layout::row_major)
    {
        call_cutlass<matrix_layout::row_major, matrix_layout::row_major>(queries, points, dist);
    }
    else if (args_.queries_layout == matrix_layout::row_major &&
             args_.points_layout == matrix_layout::column_major)
    {
        call_cutlass<matrix_layout::row_major, matrix_layout::column_major>(queries, points, dist);
    }
    else if (args_.queries_layout == matrix_layout::column_major &&
             args_.points_layout == matrix_layout::row_major)
    {
        call_cutlass<matrix_layout::column_major, matrix_layout::row_major>(queries, points, dist);
    }
    else if (args_.queries_layout == matrix_layout::column_major &&
             args_.points_layout == matrix_layout::column_major)
    {
        call_cutlass<matrix_layout::column_major, matrix_layout::column_major>(queries, points,
                                                                               dist);
    }
}
