#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/distance/stub_distance.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/raft_warpsort.hpp"

TEMPLATE_TEST_CASE("RAFT selection returns labeled minima from dense rows", "[raft-selection]",
                   air_topk, raft_warpsort)
{
    // Cover the copy path, one-block selection, and multi-block selection.
    struct shape
    {
        std::size_t points, queries, k;
    };

    const shape shapes[] = {{37, 3, 37},     {199, 7, 1},      {199, 7, 32},
                            {16384, 3, 256}, {262147, 1, 128}, {32771, 7, 32}};
    TestType algorithm;
    for (const auto& shape : shapes)
    {
        for (std::size_t unique : {0u, 5u})
        {
            CAPTURE(shape.points, shape.queries, shape.k, unique);
            knn_args args{};
            args.point_count = shape.points;
            args.query_count = shape.queries;
            args.k = shape.k;
            auto distance = std::make_unique<stub_distance>(42, unique);
            auto* input = distance.get();
            algorithm.set_dist_impl(std::move(distance));
            algorithm.initialize(args);
            // Repeated selections exercise workspace reuse with changed input.
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                algorithm.distances();
                algorithm.selection();
                const auto actual = algorithm.finish();
                const auto matrix = input->matrix_cpu();
                for (std::size_t row = 0; row < shape.queries; ++row)
                {
                    const auto* first = matrix.data() + row * shape.points;
                    std::vector<float> expected(first, first + shape.points);
                    std::sort(expected.begin(), expected.end());
                    std::vector<float> values;
                    std::vector<std::int32_t> labels;
                    for (std::size_t i = 0; i < shape.k; ++i)
                    {
                        const auto& pair = actual[row * shape.k + i];
                        REQUIRE(pair.index >= 0);
                        REQUIRE(static_cast<std::size_t>(pair.index) < shape.points);
                        REQUIRE(pair.distance == first[pair.index]);
                        values.push_back(pair.distance);
                        labels.push_back(pair.index);
                    }
                    // Radix selection returns an unordered top-k set; ties may select any labels.
                    std::sort(values.begin(), values.end());
                    expected.resize(shape.k);
                    REQUIRE(values == expected);
                    std::sort(labels.begin(), labels.end());
                    REQUIRE(std::adjacent_find(labels.begin(), labels.end()) == labels.end());
                }
            }
        }
    }
}
