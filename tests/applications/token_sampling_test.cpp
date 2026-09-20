#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/applications/token_sampling.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/grid_select.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace
{
std::vector<float> logits_for(std::size_t batch, std::size_t vocabulary)
{
    std::vector<float> result(batch * vocabulary);
    for (std::size_t row = 0; row < batch; ++row)
        for (std::size_t j = 0; j < vocabulary; ++j)
            result[row * vocabulary + j] =
                static_cast<float>(static_cast<int>((j * 37 + row * 11) % 509) - 254) / 13.0f;
    return result;
}

void check_output(applications::token_sampling& operation, std::span<const float> logits,
                  std::size_t batch, std::size_t vocabulary, std::size_t k, float temperature,
                  std::span<const float> expected, std::uint64_t seed, std::uint64_t draw)
{
    std::vector<applications::sampled_token> actual(batch);
    std::vector<std::int32_t> ids(batch * k);
    std::vector<float> probabilities(batch * k);
    operation.download(actual);
    operation.download_distribution(ids, probabilities);
    cuda_stream::make_default().sync();
    REQUIRE_NOTHROW(applications::verify_sampling(logits, batch, vocabulary, k, temperature,
                                                  expected, ids, probabilities, actual, seed,
                                                  draw));
}
} // namespace

TEMPLATE_TEST_CASE("Sampling backends preserve batched supports across updates and masks",
                   "[applications][sampling]", bits_knn, bits_prefetch_knn, single_query_bits,
                   air_topk, grid_select, block_select)
{
    struct shape
    {
        std::size_t batch, vocabulary, k;
    };

    for (const auto [batch, vocabulary, k] :
         {shape{3, 37, 32}, shape{5, 2053, 128}, shape{2, 1024, 1024}})
    {
        CAPTURE(batch, vocabulary, k);
        constexpr float temperature = 0.7f;
        applications::token_sampling operation(batch, vocabulary, k, temperature);
        TestType selector;
        knn_args args{};
        args.point_count = vocabulary;
        args.query_count = batch;
        args.k = k;
        args.deg = 32;
        args.selection_block_size = 128;
        args.items_per_thread = {1, 1, 1};
        if constexpr (std::is_base_of_v<bits_knn, TestType> ||
                      std::is_same_v<TestType, single_query_bits>)
        {
            args.selection_block_size = 512;
            args.items_per_thread[0] = std::is_same_v<TestType, single_query_bits> ? 4 : 7;
        }
        selector.set_dist_impl(std::make_unique<precomputed_scores>(operation.scores()));
        selector.initialize(args);
        for (unsigned distribution = 0; distribution < 3; ++distribution)
        {
            CAPTURE(distribution);
            auto logits = logits_for(batch, vocabulary);
            if (distribution == 1)
                for (std::size_t row = 0; row < batch; ++row)
                    for (std::size_t j = 0; j < vocabulary; ++j)
                        logits[row * vocabulary + j] =
                            j < k ? static_cast<float>(j % 3)
                                  : -std::numeric_limits<float>::infinity();
            if (distribution == 2)
                std::fill(logits.begin(), logits.end(), -3.0f);
            const auto expected =
                applications::reference_sampling_topk(logits, batch, vocabulary, k);
            operation.upload(logits);
            for (std::uint64_t draw = 0; draw < 3; ++draw)
            {
                operation.transform();
                selector.selection();
                operation.output(selector.out_dist_gpu(), selector.out_label_gpu(), 7193, draw);
                check_output(operation, logits, batch, vocabulary, k, temperature, expected, 7193,
                             draw);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Sampling supports arbitrary k and irregular rows", "[applications][sampling]",
                   bits_knn, bits_prefetch_knn, single_query_bits, air_topk, grid_select)
{
    constexpr std::size_t batch = 3, vocabulary = 2053;
    const auto logits = logits_for(batch, vocabulary);
    for (const std::size_t k : {1u, 7u, 33u, 1025u, 2048u})
    {
        CAPTURE(k);
        applications::token_sampling operation(batch, vocabulary, k, 1.0f);
        TestType selector;
        knn_args args{};
        args.point_count = vocabulary;
        args.query_count = batch;
        args.k = k;
        args.deg = 32;
        args.selection_block_size = 512;
        args.items_per_thread = {4, 1, 1};
        selector.set_dist_impl(std::make_unique<precomputed_scores>(operation.scores()));
        selector.initialize(args);
        operation.upload(logits);
        operation.transform();
        selector.selection();
        operation.output(selector.out_dist_gpu(), selector.out_label_gpu(), 13, 7);
        const auto expected = applications::reference_sampling_topk(logits, batch, vocabulary, k);
        check_output(operation, logits, batch, vocabulary, k, 1.0f, expected, 13, 7);
    }
}

TEST_CASE("Sampling softmax handles extreme logits and temperatures without reordering",
          "[applications][sampling]")
{
    const auto maximum = std::numeric_limits<float>::max();
    const std::vector<float> logits = {maximum, -maximum, 0.0f, 1.0f, -1.0f,
                                       1.0f,    1.0f,     1.0f, 1.0f, 1.0f};
    const std::vector<std::int32_t> ids = {1, 3, 0, 4, 2, 4, 3, 2, 1, 0};
    std::vector<float> scores(ids.size());
    for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t j = 0; j < 5; ++j)
            scores[row * 5 + j] = -logits[row * 5 + ids[row * 5 + j]];
    const auto expected = applications::reference_sampling_topk(logits, 2, 5, 5);
    cuda_array<float, 2> selected_scores{{2, 5}};
    cuda_array<std::int32_t, 2> selected_ids{{2, 5}};
    cuda_stream::make_default()
        .copy_to_gpu_async(selected_scores.view(), scores.data())
        .copy_to_gpu_async(selected_ids.view(), ids.data());
    for (float temperature : {std::numeric_limits<float>::denorm_min(), 0.01f, 1.0f, maximum})
    {
        CAPTURE(temperature);
        applications::token_sampling operation(2, 5, 5, temperature);
        operation.upload(logits);
        operation.output(selected_scores.view(), selected_ids.view(), 53, 97);
        check_output(operation, logits, 2, 5, 5, temperature, expected, 53, 97);
        std::vector<std::int32_t> actual_ids(10);
        std::vector<float> probabilities(10);
        operation.download_distribution(actual_ids, probabilities);
        cuda_stream::make_default().sync();
        REQUIRE(actual_ids == ids);
    }
}

TEST_CASE("Sampling probabilities and seeded draws have the expected empirical frequencies",
          "[applications][sampling]")
{
    // Independent sequence counters provide 32,768 categorical draws per invocation.
    constexpr std::size_t batch = 32768, vocabulary = 3;
    std::vector<float> logits(batch * vocabulary), scores(batch * vocabulary);
    std::vector<std::int32_t> ids(batch * vocabulary);
    const float values[] = {0.0f, std::log(2.0f), std::log(7.0f)};
    for (std::size_t row = 0; row < batch; ++row)
        for (std::size_t j = 0; j < vocabulary; ++j)
        {
            logits[row * vocabulary + j] = values[j];
            scores[row * vocabulary + j] = -values[j];
            ids[row * vocabulary + j] = static_cast<std::int32_t>(j);
        }
    applications::token_sampling operation(batch, vocabulary, vocabulary, 1.0f);
    operation.upload(logits);
    cuda_array<float, 2> selected_scores{{batch, vocabulary}};
    cuda_array<std::int32_t, 2> selected_ids{{batch, vocabulary}};
    cuda_stream::make_default()
        .copy_to_gpu_async(selected_scores.view(), scores.data())
        .copy_to_gpu_async(selected_ids.view(), ids.data());
    const auto expected =
        applications::reference_sampling_topk(logits, batch, vocabulary, vocabulary);
    std::vector<applications::sampled_token> first(batch), again(batch);
    for (std::uint64_t draw = 0; draw < 3; ++draw)
    {
        operation.output(selected_scores.view(), selected_ids.view(), 9137, draw);
        check_output(operation, logits, batch, vocabulary, vocabulary, 1.0f, expected, 9137, draw);
        operation.download(first);
        operation.output(selected_scores.view(), selected_ids.view(), 9137, draw);
        operation.download(again);
        cuda_stream::make_default().sync();
        std::array<std::size_t, 3> counts{};
        for (std::size_t row = 0; row < batch; ++row)
        {
            REQUIRE(first[row].token == again[row].token);
            REQUIRE(first[row].probability == again[row].probability);
            ++counts[first[row].token];
        }
        for (std::size_t j = 0; j < vocabulary; ++j)
        {
            const double p = j == 0 ? 0.1 : j == 1 ? 0.2 : 0.7;
            const double sigma = std::sqrt(batch * p * (1.0 - p));
            REQUIRE(std::abs(static_cast<double>(counts[j]) - batch * p) < 6.0 * sigma);
        }
    }
}

TEST_CASE("Sampling rejects invalid inputs and corrupt output", "[applications][sampling-host]")
{
    const std::vector<float> logits{0, 0, 0, -1};
    REQUIRE_NOTHROW(applications::validate_logits(logits, 1, 4, 3, 1.0f));
    for (float invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        auto bad = logits;
        bad[0] = invalid;
        REQUIRE_THROWS_AS(applications::validate_logits(bad, 1, 4, 3, 1.0f), std::invalid_argument);
    }
    auto masked = logits;
    masked[0] = masked[1] = -std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(applications::validate_logits(masked, 1, 4, 3, 1.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_logits(logits, 2, 4, 3, 1.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_logits(logits, 0, 4, 3, 1.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_logits(logits, 1, 4, 0, 1.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_logits(logits, 1, 4, 5, 1.0f), std::invalid_argument);
    for (float temperature : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()})
        REQUIRE_THROWS_AS(applications::validate_logits(logits, 1, 4, 3, temperature),
                          std::invalid_argument);
    const auto expected = applications::reference_sampling_topk(logits, 1, 4, 3);
    const auto chosen = static_cast<std::size_t>(applications::sampling_uniform(17, 9, 0) * 3);
    for (int error = -1; error < 7; ++error)
    {
        CAPTURE(error);
        std::vector<std::int32_t> ids{0, 1, 2};
        std::vector<float> probabilities(3, 1.0f / 3.0f);
        std::vector<applications::sampled_token> samples{
            {static_cast<std::int32_t>(chosen), 1.0f / 3.0f}};
        if (error == 0)
            ids[0] = -1;
        if (error == 1)
            ids[0] = ids[1];
        if (error == 2)
            ids[0] = 3;
        if (error == 3)
            probabilities[0] = 0.5f;
        if (error == 4)
            samples[0].token = static_cast<std::int32_t>((chosen + 1) % 3);
        if (error == 5)
            samples[0].probability = 0.5f;
        if (error == 6)
            probabilities.pop_back();
        if (error < 0)
            REQUIRE_NOTHROW(applications::verify_sampling(logits, 1, 4, 3, 1.0f, expected, ids,
                                                          probabilities, samples, 17, 9));
        else
            REQUIRE_THROWS(applications::verify_sampling(logits, 1, 4, 3, 1.0f, expected, ids,
                                                         probabilities, samples, 17, 9));
    }
}

TEST_CASE("Sampling detects invalid selection buffers without unsafe GPU reads",
          "[applications][sampling]")
{
    REQUIRE_THROWS_AS((applications::token_sampling{0, 4, 1, 1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::token_sampling{1, 4, 5, 1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::token_sampling{1, 4096, 2049, 1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::token_sampling{1, 4, 1, 0.0f}), std::invalid_argument);
    applications::token_sampling operation(1, 4, 3, 1.0f);
    REQUIRE_THROWS_AS(operation.upload({}), std::invalid_argument);
    REQUIRE_THROWS_AS(operation.download({}), std::invalid_argument);
    REQUIRE_THROWS_AS(operation.download_distribution({}, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(operation.output({}, {}, 0, 0), std::invalid_argument);
    const std::vector<float> logits{1, 2, 3, -std::numeric_limits<float>::infinity()};
    operation.upload(logits);
    cuda_array<float, 2> selected_scores{{1, 3}};
    cuda_array<std::int32_t, 2> selected_ids{{1, 3}};
    for (int error = 0; error < 5; ++error)
    {
        CAPTURE(error);
        std::vector<std::int32_t> ids{0, 1, 2};
        std::vector<float> scores{-1, -2, -3};
        if (error == 0)
            ids[0] = -1;
        if (error == 1)
            ids[0] = std::numeric_limits<std::int32_t>::max();
        if (error == 2)
            scores[0] = 0;
        if (error == 3)
        {
            ids[0] = 3;
            scores[0] = std::numeric_limits<float>::infinity();
        }
        if (error == 4)
            scores[0] = std::numeric_limits<float>::quiet_NaN();
        cuda_stream::make_default()
            .copy_to_gpu_async(selected_scores.view(), scores.data())
            .copy_to_gpu_async(selected_ids.view(), ids.data());
        operation.output(selected_scores.view(), selected_ids.view(), 0, 0);
        std::vector<applications::sampled_token> actual(1);
        operation.download(actual);
        cuda_stream::make_default().sync();
        if (error == 2)
        {
            std::vector<std::int32_t> selected(3);
            std::vector<float> probabilities(3);
            operation.download_distribution(selected, probabilities);
            cuda_stream::make_default().sync();
            const auto expected = applications::reference_sampling_topk(logits, 1, 4, 3);
            REQUIRE_THROWS(applications::verify_sampling(logits, 1, 4, 3, 1.0f, expected, selected,
                                                         probabilities, actual, 0, 0));
        }
        else
        {
            REQUIRE(actual[0].token == -1);
            REQUIRE(std::isnan(actual[0].probability));
        }
    }
}
