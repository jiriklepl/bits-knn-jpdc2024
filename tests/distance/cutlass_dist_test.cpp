#include <catch2/catch_test_macros.hpp>

#include "bits/distance/cutlass_distance.hpp"

#include "dist_test.hpp"

TEST_CASE("Compute distance using CUTLASS with custom euclidean operator", "[cutlass]")
{
    knn_args args;
    args.point_count = 32;
    args.query_count = 32;
    args.dim = 16;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS row, row", "[cutlass]")
{
    knn_args args;
    args.point_count = 3;
    args.query_count = 4;
    args.dim = 2;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS column, row", "[cutlass]")
{
    knn_args args;
    args.point_count = 3;
    args.query_count = 4;
    args.dim = 2;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::column_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS row, column", "[cutlass]")
{
    knn_args args;
    args.point_count = 3;
    args.query_count = 4;
    args.dim = 2;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::column_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS column, column", "[cutlass]")
{
    knn_args args;
    args.point_count = 3;
    args.query_count = 4;
    args.dim = 2;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::column_major;
    args.queries_layout = matrix_layout::column_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS for large matrix", "[cutlass]")
{
    knn_args args;
    args.point_count = 1024;
    args.query_count = 256;
    args.dim = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}

TEST_CASE("Compute distance using CUTLASS with transposed points", "[cutlass]")
{
    knn_args args;
    args.point_count = 8;
    args.query_count = 16;
    args.dim = 32;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::column_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<cutlass_distance> test{args};
    test.run();
}
