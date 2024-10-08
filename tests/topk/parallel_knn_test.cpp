#include <catch2/catch_test_macros.hpp>

#include "bits/topk/parallel_knn.hpp"

#include "knn_test.hpp"

TEST_CASE("Parallel kNN on the CPU", "[parallel_knn]")
{
    knn_args args;
    args.dim = 10;
    args.k = 8;
    args.selection_block_size = 128;
    args.point_count = 128;
    args.query_count = 32;
    args.dist_layout = matrix_layout::row_major;

    test_case<parallel_knn, eigen_distance> test{args};
    test.run();
}
