#include <catch2/catch_test_macros.hpp>

#include "bits/topk/singlepass/bits_knn.hpp"

#include "knn_test.hpp"

TEST_CASE("Partial bitonic with top k list in registers, k = 128, n = 128", "[bits]")
{
    knn_args args;
    args.dim = 10;
    args.k = 128;
    args.selection_block_size = 128;
    args.point_count = 128;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    {
        args.items_per_thread[0] = 1; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }

    {
        args.items_per_thread[0] = 7; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }

    {
        args.items_per_thread[0] = 16; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }
}

TEST_CASE("Partial bitonic with top k list in registers, k = 256, n = 1024", "[bits]")
{
    knn_args args;
    args.dim = 10;
    args.k = 256;
    args.selection_block_size = 128;
    args.point_count = 1024;
    args.query_count = 1;
    args.dist_layout = matrix_layout::row_major;

    {
        args.items_per_thread[0] = 1; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }

    {
        args.items_per_thread[0] = 7; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }

    {
        args.items_per_thread[0] = 16; // batch size for loading from global memory
        test_case<bits_knn> test{args};
        test.run();
    }
}
