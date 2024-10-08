#include <catch2/catch_test_macros.hpp>

#include "bits/topk/singlepass/fused_knn.hpp"

#include "knn_test.hpp"

TEST_CASE("Find the nearest neighbor using the fused kernel", "[fused]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 32;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel, k = 32, n = 128", "[fused]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 128;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel with multiple blocks", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 128;
    args.query_count = 128;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 64", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 64;
    args.selection_block_size = 128;
    args.point_count = 64;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 2", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 2;
    args.selection_block_size = 128;
    args.point_count = 256;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 4", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 4;
    args.selection_block_size = 128;
    args.point_count = 256;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 8", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 8;
    args.selection_block_size = 128;
    args.point_count = 256;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 16", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 16;
    args.selection_block_size = 128;
    args.point_count = 256;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}

TEST_CASE("Find the nearest neighbor using the fused kernel for k = 128", "[fused]")
{
    knn_args args;
    args.dim = 16;
    args.k = 128;
    args.selection_block_size = 128;
    args.point_count = 1024;
    args.query_count = 32;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    test_case<fused_regs_knn> test{args};
    test.run();
}
