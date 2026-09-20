#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/cuda_stream.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/score_validation.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace
{

// Accept any valid cutoff-tie subset, while checking the value/index association.
void require_topk(const std::vector<float>& input, const std::vector<distance_pair>& actual,
                  std::size_t rows, std::size_t columns, std::size_t k)
{
    REQUIRE(actual.size() == rows * k);
    for (std::size_t row = 0; row < rows; ++row)
    {
        const auto first = input.begin() + row * columns;
        std::vector<float> expected(first, first + columns);
        std::sort(expected.begin(), expected.end());
        expected.resize(k);
        std::vector<float> values;
        std::vector<std::int32_t> labels;
        for (std::size_t i = 0; i < k; ++i)
        {
            const auto& pair = actual[row * k + i];
            REQUIRE(pair.index >= 0);
            REQUIRE(static_cast<std::size_t>(pair.index) < columns);
            REQUIRE(std::isfinite(pair.distance));
            REQUIRE(pair.distance == first[pair.index]);
            values.push_back(pair.distance);
            labels.push_back(pair.index);
        }
        std::sort(values.begin(), values.end());
        REQUIRE(values == expected);
        std::sort(labels.begin(), labels.end());
        REQUIRE(std::adjacent_find(labels.begin(), labels.end()) == labels.end());
    }
}

knn_args score_args(std::size_t rows, std::size_t columns, std::size_t k, std::size_t degree)
{
    knn_args args{};
    args.point_count = columns;
    args.query_count = rows;
    args.k = k;
    args.deg = degree;
    args.selection_block_size = 128;
    args.items_per_thread = {1, 1, 1};
    args.dist_layout = matrix_layout::row_major;
    return args;
}

// No point/query buffers exist for this provider, even after prepare().
class inspect_scores : public precomputed_scores
{
public:
    using precomputed_scores::precomputed_scores;

    bool has_owned_device_storage() const
    {
        return points_gpu_.view().data() || queries_gpu_.view().data() || dist_gpu_.view().data();
    }
};

} // namespace

TEST_CASE("Precomputed scores borrow a validated dense matrix", "[scores]")
{
    cuda_array<float, 2> storage{{2, 37}};
    auto args = score_args(2, 37, 32, 1);
    inspect_scores provider(storage.view());
    provider.prepare(args);
    provider.compute();
    REQUIRE(provider.matrix_gpu().data() == storage.view().data());
    REQUIRE_FALSE(provider.has_owned_device_storage());
    REQUIRE(provider.transfer_seconds() == 0);

    args.point_count = 36;
    REQUIRE_THROWS_AS(provider.prepare(args), std::invalid_argument);
    args.point_count = 37;
    args.dist_layout = matrix_layout::column_major;
    REQUIRE_THROWS_AS(provider.prepare(args), std::invalid_argument);
    REQUIRE_THROWS_AS(precomputed_scores(array_view<float, 2>{}), std::invalid_argument);
    REQUIRE_THROWS_AS(
        precomputed_scores(array_view<float, 2>{storage.view().data(), {2, 37}, {38, 1}}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        precomputed_scores(array_view<float, 2>{storage.view().data(), {2, 37}, {37, 2}}),
        std::invalid_argument);

    // Destroying the provider does not free the caller's storage.
    {
        precomputed_scores temporary(storage.view());
        temporary.prepare(score_args(2, 37, 32, 1));
    }
    std::vector<float> input(74, -3.0f);
    cuda_stream::make_default().copy_to_gpu_async(storage.view(), input.data()).sync();
    const auto host = provider.matrix_cpu();
    REQUIRE(std::equal(input.begin(), input.end(), host.data()));
}

TEMPLATE_TEST_CASE("Selection backends consume changing application scores", "[scores]", bits_knn,
                   bits_prefetch_knn, single_query_bits, air_topk, block_select)
{
    struct shape
    {
        std::size_t rows, columns, k, degree;
    };

    const shape shapes[] = {{3, 128, 32, 4},   {3, 131, 32, 4},      {2, 37, 32, 7},
                            {3, 32, 32, 32},   {1, 262147, 128, 64}, {8, 50257, 64, 1},
                            {64, 50257, 32, 1}};
    TestType algorithm;
    for (const auto& shape : shapes)
    {
        CAPTURE(shape.rows, shape.columns, shape.k, shape.degree);
        // Reinitialize the same algorithm to check workspace resizing and stale offsets.
        cuda_array<float, 2> storage{{shape.rows, shape.columns}};
        algorithm.set_dist_impl(std::make_unique<precomputed_scores>(storage.view()));
        algorithm.initialize(score_args(shape.rows, shape.columns, shape.k, shape.degree));
        std::vector<float> values(shape.rows * shape.columns);
        for (int distribution = 0; distribution < 4; ++distribution)
        {
            CAPTURE(distribution);
            for (std::size_t row = 0; row < shape.rows; ++row)
            {
                for (std::size_t col = 0; col < shape.columns; ++col)
                {
                    float value = static_cast<float>(col) - shape.columns / 2.0f;
                    if (distribution == 1)
                        value = static_cast<float>(col % 7) - 3.0f;
                    if (distribution == 2)
                        value = 0.0f;
                    if (distribution == 3 && col >= shape.k && col % 5 == 0)
                        value = std::numeric_limits<float>::infinity();
                    values[row * shape.columns + col] = value;
                }
                std::mt19937 random(42 + row + distribution);
                auto first = values.begin() + row * shape.columns;
                std::shuffle(first, first + shape.columns, random);
            }
            cuda_stream::make_default().copy_to_gpu_async(storage.view(), values.data()).sync();
            validate_score_values(values, shape.rows, shape.columns, shape.k);
            algorithm.distances();
            algorithm.selection();
            require_topk(values, algorithm.finish(), shape.rows, shape.columns, shape.k);
        }
    }
}

TEST_CASE("BITS split-and-merge supports arbitrary k and reinitialization to one partition",
          "[scores]")
{
    single_query_bits algorithm;
    for (std::size_t degree : {4u, 1u, 17u})
    {
        for (std::size_t k : {1u, 7u, 17u})
        {
            CAPTURE(degree, k);
            cuda_array<float, 2> storage{{3, 17}};
            std::vector<float> input(51);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = 30.0f - i;
            cuda_stream::make_default().copy_to_gpu_async(storage.view(), input.data()).sync();
            algorithm.set_dist_impl(std::make_unique<precomputed_scores>(storage.view()));
            algorithm.initialize(score_args(3, 17, k, degree));
            algorithm.selection();
            require_topk(input, algorithm.finish(), 3, 17, k);
        }
    }
}

TEST_CASE("Score transformations preserve largest values and signed magnitudes", "[scores]")
{
    const std::vector<float> original{3, -8, 1, 8, -2, 0, -7, 4, -1, 6, 2};
    for (bool magnitude : {false, true})
    {
        CAPTURE(magnitude);
        std::vector<float> scores;
        for (float value : original)
            scores.push_back(magnitude ? -std::abs(value) : -value);
        cuda_array<float, 2> storage{{1, scores.size()}};
        cuda_stream::make_default().copy_to_gpu_async(storage.view(), scores.data()).sync();
        single_query_bits algorithm;
        algorithm.set_dist_impl(std::make_unique<precomputed_scores>(storage.view()));
        algorithm.initialize(score_args(1, scores.size(), 3, 4));
        algorithm.selection();
        const auto selected = algorithm.finish();
        require_topk(scores, selected, 1, scores.size(), 3);
        for (const auto& pair : selected)
        {
            const float gathered = original[pair.index];
            REQUIRE((magnitude ? -std::abs(gathered) : -gathered) == pair.distance);
        }
    }
}

TEST_CASE("BITS rejects invalid partition degrees before touching input", "[scores]")
{
    single_query_bits algorithm;
    REQUIRE_THROWS_AS(algorithm.initialize(score_args(1, 17, 3, 0)), std::invalid_argument);
    REQUIRE_THROWS_AS(algorithm.initialize(score_args(1, 17, 3, 18)), std::invalid_argument);
}

TEST_CASE("BITS split selection bounds uneven and pitched input rows", "[scores]")
{
    class pitched_scores : public cuda_distance
    {
    public:
        explicit pitched_scores(array_view<float, 2> scores) : scores_(scores) {}

        void prepare(const knn_args& args) override { args_ = args; }

        void compute() override {}

        array_view<float, 2> matrix_gpu() const override { return scores_; }

        std::string name() const override { return "pitched-test-scores"; }

    private:
        array_view<float, 2> scores_;
    };

    struct shape
    {
        std::size_t rows, columns, k, degree, padding;
    };

    // Ceil-sized partitions leave one empty tail partition for 35/8 and five for 133/32.
    // Their nonempty partitions also contain fewer candidates than k.
    const shape shapes[] = {{3, 128, 32, 1, 9},
                            {3, 128, 32, 4, 9},
                            {3, 35, 32, 8, 0},
                            {3, 35, 32, 8, 9},
                            {4, 133, 65, 32, 9},
                            // More than one register buffer per partition, with unequal row ends.
                            {3, 4099, 65, 3, 22}};
    single_query_bits algorithm;
    for (const auto& shape : shapes)
    {
        CAPTURE(shape.rows, shape.columns, shape.k, shape.degree, shape.padding);
        const auto stride = shape.columns + shape.padding;
        cuda_array<float, 2> storage{{shape.rows, stride}};
        const array_view<float, 2> scores{
            storage.view().data(), {shape.rows, shape.columns}, {stride, 1}};
        algorithm.set_dist_impl(std::make_unique<pitched_scores>(scores));
        algorithm.initialize(score_args(shape.rows, shape.columns, shape.k, shape.degree));

        // Padding would outrank every valid candidate if a partition read beyond its row.
        std::vector<float> physical(shape.rows * stride, std::numeric_limits<float>::lowest());
        std::vector<float> input(shape.rows * shape.columns);
        for (bool reverse : {false, true})
        {
            CAPTURE(reverse);
            for (std::size_t row = 0; row < shape.rows; ++row)
            {
                for (std::size_t col = 0; col < shape.columns; ++col)
                {
                    // Distinct row ranges expose cross-row reads. Reversing both orders makes
                    // the short final partition essential and detects stale selection inputs.
                    const float value = reverse ? -10000.0f * (row + 1) + (shape.columns - col)
                                                : 10000.0f * row + col;
                    input[row * shape.columns + col] = value;
                    physical[row * stride + col] = value;
                }
            }
            cuda_stream::make_default().copy_to_gpu_async(storage.view(), physical.data()).sync();
            algorithm.selection();
            require_topk(input, algorithm.finish(), shape.rows, shape.columns, shape.k);
        }
    }
}
