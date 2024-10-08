#include <catch2/catch_test_macros.hpp>

#include "bits/distance/magma_distance.hpp"

#include "dist_test.hpp"

TEST_CASE("Compute distance using the magma approach", "[magma]")
{
    knn_args args;
    args.point_count = 32;
    args.query_count = 32;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_distance> test{args};
    test.run();
}

TEST_CASE("Compute long distance matrix using the magma approach", "[magma]")
{
    knn_args args;
    args.point_count = 128;
    args.query_count = 1;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance matrix in column major order using the magma approach", "[magma]")
{
    knn_args args;
    args.point_count = 128;
    args.query_count = 1;
    args.dim = 16;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance matrix with small odd dimension in MAGMA", "[magma]")
{
    knn_args args;
    args.point_count = 32;
    args.query_count = 32;
    args.dim = 9;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::column_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_distance> test{args};
    test.run();
}

TEST_CASE("Compute matrix with partial distances using MAGMA", "[magma-part]")
{
    knn_args args;
    args.point_count = 1024;
    args.query_count = 32;
    args.dim = 32;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_partial_distance, add_query_norms> test{args};
    test.run();
}

TEST_CASE("Compute matrix with partial distances using MAGMA on a thin matrix and an odd dimension",
          "[magma-part]")
{
    knn_args args;
    args.point_count = 23 * 1024;
    args.query_count = 1;
    args.dim = 17;
    args.dist_block_size = 256;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<magma_partial_distance, add_query_norms> test{args};
    test.run();
}
