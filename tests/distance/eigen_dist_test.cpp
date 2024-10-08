#include <catch2/catch_test_macros.hpp>

#include "bits/distance/eigen_distance.hpp"

#include "dist_test.hpp"

TEST_CASE("Compute distances of low dimensional points using Eigen", "[eigen]")
{
    knn_args args;
    args.point_count = 64;
    args.query_count = 32;
    args.dim = 4;
    args.dist_block_size = 512;
    args.dist_layout = matrix_layout::row_major;
    args.points_layout = matrix_layout::row_major;
    args.queries_layout = matrix_layout::row_major;

    dist_test<eigen_distance, add_query_norms> test{args};
    test.run();
}
