#ifndef BITS_KNN_ARGS_HPP_
#define BITS_KNN_ARGS_HPP_

#include <array>
#include <cstddef>

#include "bits/layout.hpp"

struct knn_args
{
    // matrix of points
    const float* points;
    // matrix of queries
    const float* queries;
    // number of point vectors
    std::size_t point_count;
    // number of query vectors
    std::size_t query_count;
    // dimension of vectors in the points matrix and the query matrix
    std::size_t dim;
    // layout of the points matrix
    matrix_layout points_layout;
    // layout of the query matrix
    matrix_layout queries_layout;
    // layout of the output distance matrix
    matrix_layout dist_layout;
    // CUDA thread block size for the distance kernel (some implementations will ignore this option)
    std::size_t dist_block_size;
    // CUDA thread block size for k-selection (some implementations will ignore this option)
    std::size_t selection_block_size;
    // number of nearest neighbors
    std::size_t k;
    // number of items per thread
    std::array<std::size_t, 3> items_per_thread;
    // degree of parallelism for single-query problems
    std::size_t deg;
};

#endif // BITS_KNN_ARGS_HPP_
