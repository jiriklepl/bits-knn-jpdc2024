#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/applications/gradient_compression.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/grid_select.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace
{
std::vector<float> gradient(std::size_t n)
{
    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = static_cast<float>(int((i * 37) % 137) - 68) * 0.03125f;
    return result;
}

template <class Selector>
void configure(Selector& selector, applications::gradient_compression& compressor, std::size_t n,
               std::size_t k, std::size_t degree)
{
    knn_args args{};
    args.point_count = n;
    args.query_count = 1;
    args.k = k;
    args.deg = degree;
    args.selection_block_size = 128;
    args.items_per_thread = {1, 1, 1};
    if constexpr (std::is_base_of_v<bits_knn, Selector> ||
                  std::is_same_v<Selector, single_query_bits>)
    {
        args.selection_block_size = 512;
        args.items_per_thread[0] = std::is_same_v<Selector, single_query_bits> ? 4 : 7;
    }
    selector.set_dist_impl(std::make_unique<precomputed_scores>(compressor.scores()));
    selector.initialize(args);
}

template <class Selector>
std::vector<applications::gradient_entry>
compress(Selector& selector, applications::gradient_compression& compressor, std::size_t k)
{
    compressor.transform();
    selector.selection();
    compressor.output(selector.out_dist_gpu(), selector.out_label_gpu());
    std::vector<applications::gradient_entry> actual(k);
    compressor.download(actual);
    cuda_stream::make_default().sync();
    return actual;
}
} // namespace

TEMPLATE_TEST_CASE("Gradient compression preserves signed values across repeated updates",
                   "[applications][gradient]", bits_knn, bits_prefetch_knn, single_query_bits,
                   air_topk, grid_select, block_select)
{
    for (const auto [n, k] :
         {std::pair<std::size_t, std::size_t>{37, 32}, {131, 64}, {1024, 1024}, {8197, 128}})
    {
        CAPTURE(n, k);
        auto input = gradient(n);
        applications::gradient_compression compressor(n, k);
        TestType selector;
        configure(selector, compressor, n, k, 32);
        for (unsigned distribution = 0; distribution < 4; ++distribution)
        {
            CAPTURE(distribution);
            if (distribution == 1)
                for (std::size_t i = 0; i < n; ++i)
                    input[i] = i % 2 ? -1.0f : 1.0f;
            if (distribution == 2)
                for (std::size_t i = 0; i < n; ++i)
                    input[i] = i % 2 ? -0.0f : 0.0f;
            if (distribution == 3)
                for (std::size_t i = 0; i < n; ++i)
                    input[i] = -float(i + 1);
            const auto expected = applications::reference_gradient_topk(input, k);
            compressor.upload(input);
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                const auto actual = compress(selector, compressor, k);
                REQUIRE_NOTHROW(applications::verify_gradient_topk(input, expected, actual));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Gradient selectors handle arbitrary k and irregular tensor lengths",
                   "[applications][gradient]", bits_knn, bits_prefetch_knn, single_query_bits,
                   air_topk, grid_select)
{
    for (const auto [n, k] : {std::pair<std::size_t, std::size_t>{1, 1},
                              {37, 7},
                              {131, 33},
                              {2053, 1025},
                              {2053, 2048}})
    {
        CAPTURE(n, k);
        const auto input = gradient(n);
        const auto expected = applications::reference_gradient_topk(input, k);
        applications::gradient_compression compressor(n, k);
        compressor.upload(input);
        TestType selector;
        configure(selector, compressor, n, k, std::min<std::size_t>(n, 32));
        const auto actual = compress(selector, compressor, k);
        REQUIRE_NOTHROW(applications::verify_gradient_topk(input, expected, actual));
    }
}

TEST_CASE("Gradient magnitude transform handles FP32 extremes and signed zero",
          "[applications][gradient]")
{
    const std::vector<float> input = {0.0f,
                                      -0.0f,
                                      1.0f,
                                      -1.0f,
                                      std::numeric_limits<float>::denorm_min(),
                                      -std::numeric_limits<float>::denorm_min(),
                                      std::numeric_limits<float>::min(),
                                      -std::numeric_limits<float>::min(),
                                      std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max()};
    applications::gradient_compression compressor(input.size(), input.size());
    compressor.upload(input);
    compressor.transform();
    std::vector<float> scores(input.size());
    cuda_stream::make_default().copy_from_gpu_async(scores.data(), compressor.scores()).sync();
    for (std::size_t i = 0; i < input.size(); ++i)
        REQUIRE(std::bit_cast<std::uint32_t>(scores[i]) ==
                std::bit_cast<std::uint32_t>(-std::fabs(input[i])));
}

TEST_CASE("Gradient output preserves input bits and does not require sorted selection",
          "[applications][gradient]")
{
    const std::vector<float> input = {-0.0f,
                                      0.0f,
                                      -3.0f,
                                      3.0f,
                                      -std::numeric_limits<float>::denorm_min(),
                                      std::numeric_limits<float>::max()};
    const std::int32_t labels[] = {1, 5, 4, 0, 2, 3};
    std::vector<float> scores;
    for (auto label : labels)
        scores.push_back(-std::fabs(input[label]));
    const auto k = input.size();
    applications::gradient_compression compressor(k, k);
    compressor.upload(input);
    cuda_array<float, 2> selected_scores{{1, k}};
    cuda_array<std::int32_t, 2> selected_labels{{1, k}};
    cuda_stream::make_default()
        .copy_to_gpu_async(selected_scores.view(), scores.data())
        .copy_to_gpu_async(selected_labels.view(), labels);
    compressor.output(selected_scores.view(), selected_labels.view());
    std::vector<applications::gradient_entry> actual(k);
    compressor.download(actual);
    cuda_stream::make_default().sync();
    REQUIRE_NOTHROW(applications::verify_gradient_topk(
        input, applications::reference_gradient_topk(input, k), actual));
    for (std::size_t i = 0; i < k; ++i)
        REQUIRE(actual[i].index == labels[i]);
}

TEST_CASE("Gradient reference allows cutoff ties and rejects corrupt output",
          "[applications][gradient-host]")
{
    const std::vector<float> input = {-4, 4, -3, 3, 0.0f, -0.0f};
    const auto expected = applications::reference_gradient_topk(input, 3);
    REQUIRE(expected == std::vector<float>{4, 4, 3});
    const std::vector<applications::gradient_entry> good = {{3, 3}, {0, -4}, {1, 4}};
    REQUIRE_NOTHROW(applications::verify_gradient_topk(input, expected, good));
    auto alternate = good;
    alternate[0] = {2, -3};
    REQUIRE_NOTHROW(applications::verify_gradient_topk(input, expected, alternate));
    for (int error = 0; error < 8; ++error)
    {
        CAPTURE(error);
        auto wrong = good;
        if (error == 0)
            wrong[0].index = -1;
        if (error == 1)
            wrong[0].index = input.size();
        if (error == 2)
            wrong[0].value = -wrong[0].value;
        if (error == 3)
            wrong[0] = wrong[1];
        if (error == 4)
            wrong.pop_back();
        if (error == 5)
            wrong[0] = {4, 0};
        if (error == 6)
            wrong[0].value = std::numeric_limits<float>::quiet_NaN();
        if (error == 7)
            wrong[0].value = std::numeric_limits<float>::infinity();
        REQUIRE_THROWS(applications::verify_gradient_topk(input, expected, wrong));
    }
    const std::vector<float> zeros = {-0.0f, 0.0f};
    std::vector<applications::gradient_entry> selected = {{0, -0.0f}};
    const auto zero_expected = applications::reference_gradient_topk(zeros, 1);
    REQUIRE_NOTHROW(applications::verify_gradient_topk(zeros, zero_expected, selected));
    selected[0].value = 0.0f;
    REQUIRE_THROWS(applications::verify_gradient_topk(zeros, zero_expected, selected));
    REQUIRE_THROWS(applications::verify_gradient_topk({}, {}, {}));
    REQUIRE_THROWS_AS(applications::reference_gradient_topk(input, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::reference_gradient_topk(input, 7), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_gradient({}), std::invalid_argument);
    for (float invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()})
    {
        auto bad = input;
        bad[2] = invalid;
        REQUIRE_THROWS_AS(applications::validate_gradient(bad), std::invalid_argument);
        REQUIRE_THROWS_AS(applications::reference_gradient_topk(bad, 3), std::invalid_argument);
    }
}

TEST_CASE("Gradient operator rejects invalid shapes and safely surfaces bad indices",
          "[applications][gradient]")
{
    REQUIRE_THROWS_AS((applications::gradient_compression{0, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::gradient_compression{2, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::gradient_compression{2, 3}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::gradient_compression{4096, 2049}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::gradient_compression{
                          std::size_t{std::numeric_limits<std::int32_t>::max()} + 1, 1}),
                      std::invalid_argument);
    applications::gradient_compression compressor(3, 3);
    const auto input = gradient(3);
    REQUIRE_THROWS_AS(compressor.upload({}), std::invalid_argument);
    REQUIRE_THROWS_AS(compressor.download({}), std::invalid_argument);
    REQUIRE_THROWS_AS(compressor.output({}, {}), std::invalid_argument);
    cuda_array<float, 2> scores{{1, 3}};
    cuda_array<std::int32_t, 2> indices{{1, 3}};
    REQUIRE_THROWS_AS(compressor.output({scores.view().data(), {1, 3}, {6, 2}}, indices.view()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(compressor.output(scores.view(), {indices.view().data(), {3, 1}, {1, 1}}),
                      std::invalid_argument);
    const std::int32_t bad_indices[] = {-1, 3, std::numeric_limits<std::int32_t>::max()};
    const float values[] = {-3, -2, -1};
    compressor.upload(input);
    cuda_stream::make_default()
        .copy_to_gpu_async(indices.view(), bad_indices)
        .copy_to_gpu_async(scores.view(), values);
    compressor.output(scores.view(), indices.view());
    std::vector<applications::gradient_entry> actual(3);
    compressor.download(actual);
    cuda_stream::make_default().sync();
    for (const auto& entry : actual)
        REQUIRE(entry.index == -1);
    REQUIRE_THROWS(applications::verify_gradient_topk(
        input, applications::reference_gradient_topk(input, 3), actual));
}
