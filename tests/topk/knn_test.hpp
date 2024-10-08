#ifndef KNN_TEST_HPP_
#define KNN_TEST_HPP_

#include <memory>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/uniform_generator.hpp"

#include "bits/distance/magma_distance.hpp"
#include "bits/topk/serial_knn.hpp"

template <class Algorithm, class Distance = magma_distance, bool TEST_DISTANCES = false>
struct test_case
{
    Algorithm alg;
    serial_knn serial;
    knn_args args;

    test_case(const knn_args& arguments) : args(arguments)
    {
        uniform_generator gen{42};
        auto points = gen.generate(args.point_count, args.dim);
        auto query = gen.generate(args.query_count, args.dim);

        args.points = points.data();
        args.queries = query.data();

        alg.set_dist_impl(std::make_unique<Distance>());
        alg.initialize(args);
        alg.prepare();
        alg.distances();

        serial.set_dist_impl(std::make_unique<Distance>());
        serial.initialize(args);
        serial.prepare();
        serial.distances();
    }

    void run()
    {
        alg.selection();
        alg.postprocessing();
        serial.selection();
        serial.postprocessing();
        const auto actual = alg.finish();
        const auto expected = serial.finish();

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            REQUIRE(actual[i].index == expected[i].index);
            if (TEST_DISTANCES)
            {
                REQUIRE(actual[i].distance == Catch::Approx(expected[i].distance));
            }
        }
    }
};

#endif // KNN_TEST_HPP_
