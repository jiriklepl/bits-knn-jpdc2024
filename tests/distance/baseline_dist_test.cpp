#include <catch2/catch_test_macros.hpp>

#include "bits/distance/baseline_distance.hpp"

#include "dist_test.hpp"

TEST_CASE("Compute distance using the data parallel approach", "[baseline_dist]")
{
    knn_args args;
    args.point_count = 32;
    args.query_count = 32;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<baseline_distance> test{args};
    test.run();
}

TEST_CASE("Compute long distance matrix using the data parallel approach", "[baseline_dist]")
{
    knn_args args;
    args.point_count = 128;
    args.query_count = 1;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<baseline_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance matrix in column major order using the data parallel approach",
          "[baseline_dist]")
{
    knn_args args;
    args.point_count = 128;
    args.query_count = 1;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<baseline_distance> test{args};
    test.run();
}
