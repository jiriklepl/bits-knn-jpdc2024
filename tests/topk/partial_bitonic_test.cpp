#include <catch2/catch_test_macros.hpp>

#include "bits/topk/singlepass/partial_bitonic.hpp"
#include "bits/topk/singlepass/partial_bitonic_buffered.hpp"

#include "knn_test.hpp"

TEST_CASE("Select top k using the partial bitonic algorithm", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 199;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<partial_bitonic> test{args};
    test.run();
}

TEST_CASE("Select top k using the buffered partial bitonic algorithm", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 199;
    args.query_count = 13;
    args.dist_layout = matrix_layout::row_major;

    test_case<buffered_partial_bitonic> test{args};
    test.run();
}

TEST_CASE("Select top k using the static partial bitonic algorithm", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 199;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<partial_bitonic_warp_static> test{args};
    test.run();
}

TEST_CASE("Select top k using the static buffered partial bitonic algorithm", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 32;
    args.selection_block_size = 128;
    args.point_count = 199;
    args.query_count = 13;
    args.dist_layout = matrix_layout::row_major;

    test_case<static_buffered_partial_bitonic> test{args};
    test.run();
}

TEST_CASE("Select top k from large number of points", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 256;
    args.selection_block_size = 128;
    args.point_count = 1 << 15;
    args.query_count = 13;
    args.dist_layout = matrix_layout::row_major;

    test_case<static_buffered_partial_bitonic> test{args};
    test.run();
}

TEST_CASE("Sort values using partial-bitonic-buffered-static", "[partial_bitonic]")
{
    knn_args args;
    args.dim = 10;
    args.k = 128;
    args.selection_block_size = 128;
    args.point_count = 128;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    test_case<static_buffered_partial_bitonic> test{args};
    test.run();
}
