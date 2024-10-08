#include <catch2/catch_test_macros.hpp>

#include "bits/topk/singlepass/cub_knn.hpp"

#include "knn_test.hpp"

TEST_CASE("Run cub knn on a small array", "[cub]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 32;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<cub_knn> test{args};
    test.run();
}

TEST_CASE("Run cub knn on bigger array", "[cub]")
{
    knn_args args;
    args.dim = 10;
    args.k = 128;
    args.selection_block_size = 128;
    args.point_count = 1024;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<cub_knn> test{args};
    test.run();
}

TEST_CASE("Run cub knn on a bigger array with small k", "[cub]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 256;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<cub_knn> test{args};
    test.run();
}

TEST_CASE("Run cub knn on bigger array with multiple queries", "[cub]")
{
    knn_args args;
    args.dim = 10;
    args.k = 128;
    args.selection_block_size = 128;
    args.point_count = 1024;
    args.query_count = 128;
    args.dist_layout = matrix_layout::row_major;

    test_case<cub_knn> test{args};
    test.run();
}
