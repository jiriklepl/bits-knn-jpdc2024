#include <catch2/catch_test_macros.hpp>

#include "bits/distance/tiled_distance.hpp"

#include "dist_test.hpp"

TEST_CASE("Compute distance to a single query", "[tiled_dist]")
{
    knn_args args;
    args.point_count = 251;
    args.query_count = 1;
    args.dim = 10;
    args.dist_block_size = 16;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<tiled_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance to to multiple queries", "[tiled_dist]")
{
    knn_args args;
    args.point_count = 251;
    args.query_count = 199;
    args.dim = 10;
    args.dist_block_size = 16;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<tiled_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance with high dimension", "[tiled_dist]")
{
    knn_args args;
    args.point_count = 100;
    args.query_count = 40;
    args.dim = 199;
    args.dist_block_size = 16;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<tiled_distance> test{args};
    test.run();
}

TEST_CASE("Compute distances in column major layout", "[tiled_dist]")
{
    knn_args args;
    args.point_count = 100;
    args.query_count = 40;
    args.dim = 199;
    args.dist_block_size = 16;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<tiled_distance> test{args};
    test.run();
}
