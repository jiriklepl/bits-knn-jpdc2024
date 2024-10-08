#include <catch2/catch_test_macros.hpp>

#include <array>

#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/knn_args.hpp"
#include "bits/topk/singlepass/fused_cache_knn.hpp"

#include "knn_test.hpp"

TEST_CASE("Find the nearest neighbor using the fused cache kernel", "[fused-cache]")
{
    knn_args args;
    args.dim = 8;
    args.k = 32;
    args.point_count = 1024;
    args.query_count = 32;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;
    args.items_per_thread = std::array<std::size_t, 3>{4, 4, 2};
    args.selection_block_size = 4;
    args.deg = 2;

    test_case<fused_cache_knn> test{args};
    test.run();
}

TEST_CASE("fused-cache kernel dim = 16, n = 1024", "[fused-cache]")
{
    knn_args args;
    args.dim = 16;
    args.k = 32;
    args.point_count = 1024;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;
    args.items_per_thread = std::array<std::size_t, 3>{4, 4, 2};
    args.selection_block_size = 4;
    args.deg = 2;

    test_case<fused_cache_knn> test{args};
    test.run();
}

TEST_CASE("fused-cache kernel dim = 37, n = 997, q = 43", "[fused-cache]")
{
    knn_args args;
    args.dim = 37;
    args.k = 32;
    args.point_count = 997;
    args.query_count = 43;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;
    args.items_per_thread = std::array<std::size_t, 3>{4, 8, 4};
    args.selection_block_size = 4;
    args.deg = 2;

    test_case<fused_cache_knn> test{args};
    test.run();
}

TEST_CASE("fused-cache kernel dim = 9, n = 25, q = 7", "[fused-cache]")
{
    knn_args args;
    args.dim = 9;
    args.k = 16;
    args.point_count = 25;
    args.query_count = 7;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;
    args.items_per_thread = std::array<std::size_t, 3>{4, 8, 4};
    args.selection_block_size = 4;
    args.deg = 2;

    test_case<fused_cache_knn> test{args};
    test.run();
}
