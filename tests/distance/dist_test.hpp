#ifndef DIST_TEST_HPP_
#define DIST_TEST_HPP_

#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/knn_args.hpp"
#include "bits/uniform_generator.hpp"

inline std::vector<float> compute_distances(const std::vector<float>& points,
                                            const std::vector<float>& queries,
                                            std::size_t point_count, std::size_t query_count,
                                            std::size_t dim)
{
    std::vector<float> result;
    result.reserve(query_count * point_count);

    for (std::size_t i = 0; i < query_count; ++i)
    {
        for (std::size_t j = 0; j < point_count; ++j)
        {
            float dist = 0;
            for (std::size_t k = 0; k < dim; ++k)
            {
                const auto diff = points[j * dim + k] - queries[i * dim + k];
                dist += diff * diff;
            }
            result.emplace_back(dist);
        }
    }
    return result;
}

inline std::vector<float> transpose(const float* matrix, std::size_t row_count,
                                    std::size_t col_count)
{
    std::vector<float> result(row_count * col_count);
    for (std::size_t i = 0; i < row_count; ++i)
    {
        for (std::size_t j = 0; j < col_count; ++j)
        {
            result[j * row_count + i] = matrix[i * col_count + j];
        }
    }
    return result;
}

// do not do any postprocessing on the computed distances
struct no_postprocessing
{
};

// compute and add query norms to the computed distances
struct add_query_norms
{
};

template <class Distance, class Processing = no_postprocessing>
struct dist_test
{
    Distance dist;
    knn_args args;
    std::vector<float> points;
    std::vector<float> queries;
    std::vector<float> points_trans;
    std::vector<float> queries_trans;

    dist_test(const knn_args& arguments) : args(arguments)
    {
        uniform_generator gen{42};
        points = gen.generate(args.point_count, args.dim);
        queries = gen.generate(args.query_count, args.dim);

        points_trans = transpose(points.data(), args.point_count, args.dim);
        queries_trans = transpose(queries.data(), args.query_count, args.dim);

        args.points = points.data();
        if (args.points_layout == matrix_layout::column_major)
        {
            args.points = points_trans.data();
        }

        args.queries = queries.data();
        if (args.queries_layout == matrix_layout::column_major)
        {
            args.queries = queries_trans.data();
        }
    }

    void run()
    {
        dist.prepare(args);
        dist.compute();

        // get computed distances
        const auto actual_dist = dist.matrix_cpu();
        // compute the distances using a serial algorithm
        const auto expected_dist =
            compute_distances(points, queries, args.point_count, args.query_count, args.dim);

        for (std::size_t i = 0; i < args.query_count; ++i)
        {
            for (std::size_t j = 0; j < args.point_count; ++j)
            {
                const auto actual = args.dist_layout == matrix_layout::row_major
                                        ? actual_dist(i, j)
                                        : actual_dist(j, i);
                const auto expected = expected_dist[i * args.point_count + j];
                REQUIRE(actual == Catch::Approx(expected));
            }
        }
    }
};

// specialization which adds norms of query vectors
template <class Distance>
struct dist_test<Distance, add_query_norms>
{
    Distance dist;
    knn_args args;
    std::vector<float> points;
    std::vector<float> queries;
    std::vector<float> points_trans;
    std::vector<float> queries_trans;
    std::vector<float> norm;

    dist_test(const knn_args& arguments) : args(arguments)
    {
        uniform_generator gen{42};
        points = gen.generate(args.point_count, args.dim);
        queries = gen.generate(args.query_count, args.dim);

        points_trans = transpose(points.data(), args.point_count, args.dim);
        queries_trans = transpose(queries.data(), args.query_count, args.dim);

        args.points = points.data();
        if (args.points_layout == matrix_layout::column_major)
        {
            args.points = points_trans.data();
        }

        args.queries = queries.data();
        if (args.queries_layout == matrix_layout::column_major)
        {
            args.queries = queries_trans.data();
        }

        // compute norm^2 of the query vectors
        for (std::size_t i = 0; i < args.query_count; ++i)
        {
            float result = 0;
            for (std::size_t j = 0; j < args.dim; ++j)
            {
                const auto elem = queries[i * args.dim + j];
                result += elem * elem;
            }
            norm.emplace_back(result);
        }
    }

    void run()
    {
        dist.prepare(args);
        dist.compute();

        // get computed distances
        const auto actual_dist = dist.matrix_cpu();
        // compute the distances using a serial algorithm
        const auto expected_dist =
            compute_distances(points, queries, args.point_count, args.query_count, args.dim);

        for (std::size_t i = 0; i < args.query_count; ++i)
        {
            for (std::size_t j = 0; j < args.point_count; ++j)
            {
                auto actual = args.dist_layout == matrix_layout::row_major ? actual_dist(i, j)
                                                                           : actual_dist(j, i);
                actual += norm[i];
                const auto expected = expected_dist[i * args.point_count + j];
                REQUIRE(actual == Catch::Approx(expected).epsilon(1e-3));
            }
        }
    }
};

#endif // DIST_TEST_HPP_
