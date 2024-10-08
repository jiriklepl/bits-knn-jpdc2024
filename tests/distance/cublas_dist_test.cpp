#include <catch2/catch_test_macros.hpp>

#include "bits/distance/cublas_distance.hpp"
#include "dist_test.hpp"

TEST_CASE("Computate distances of low dimensional points using cuBLAS (row, row)", "[cublas]")
{
    knn_args args;
    args.point_count = 64;
    args.query_count = 32;
    args.dim = 2;
    args.dist_block_size = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cublas_distance, add_query_norms> test{args};
    test.run();
}

TEST_CASE("Computate distances of low dimensional points using cuBLAS (row, column)", "[cublas]")
{
    knn_args args;
    args.point_count = 64;
    args.query_count = 32;
    args.dim = 2;
    args.dist_block_size = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::column_major;

    dist_test<cublas_distance, add_query_norms> test{args};
    test.run();
}

TEST_CASE("Computate distances of low dimensional points using cuBLAS (column, row)", "[cublas]")
{
    knn_args args;
    args.point_count = 64;
    args.query_count = 32;
    args.dim = 2;
    args.dist_block_size = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::column_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cublas_distance, add_query_norms> test{args};
    test.run();
}

TEST_CASE("Computate distances of low dimensional points using cuBLAS (column, column)", "[cublas]")
{
    knn_args args;
    args.point_count = 64;
    args.query_count = 32;
    args.dim = 2;
    args.dist_block_size = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::column_major;
    args.queries_layout = matrix_layout::column_major;

    dist_test<cublas_distance, add_query_norms> test{args};
    test.run();
}
