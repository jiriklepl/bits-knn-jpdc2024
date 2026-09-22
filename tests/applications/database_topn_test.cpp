#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bits/applications/database_topn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/grid_select.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace
{
applications::database_columns table(std::size_t n)
{
    applications::database_columns result;
    for (std::size_t i = 0; i < n; ++i)
    {
        result.price.push_back(static_cast<float>(int(i % 97) - 48) * 1.001f);
        result.discount.push_back(static_cast<float>(i % 11) / 100.0f);
        result.payload.push_back(static_cast<float>(i) + 0.5f);
        result.row_id.push_back((std::uint64_t{1} << 40) + i * 7);
    }
    return result;
}

std::vector<applications::database_row> reference_rows(const applications::database_columns& t)
{
    std::vector<applications::database_row> result;
    for (std::size_t i = 0; i < t.size(); ++i)
        result.push_back({t.row_id[i], applications::discounted_price(t.price[i], t.discount[i]),
                          t.payload[i], static_cast<std::int32_t>(i)});
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    return result;
}
} // namespace

TEMPLATE_TEST_CASE("Database operator returns sorted projected rows across repeated updates",
                   "[applications][database]", bits_knn, bits_prefetch_knn, single_query_bits,
                   air_topk, grid_select, block_select)
{
    struct shape
    {
        std::size_t n, k, degree;
    };

    const shape shapes[] = {{37, 32, 7}, {131, 64, 4}, {1024, 1024, 64}, {8197, 128, 64}};
    for (const auto& s : shapes)
    {
        CAPTURE(s.n, s.k, s.degree);
        auto columns = table(s.n);
        applications::database_topn query(s.n, s.k);
        TestType selector;
        knn_args args{};
        args.point_count = s.n;
        args.query_count = 1;
        args.k = s.k;
        args.deg = s.degree;
        args.selection_block_size = 128;
        args.items_per_thread = {1, 1, 1};
        if constexpr (std::is_base_of_v<bits_knn, TestType> ||
                      std::is_same_v<TestType, single_query_bits>)
        {
            args.selection_block_size = 512;
            args.items_per_thread[0] = std::is_same_v<TestType, single_query_bits> ? 4 : 7;
        }
        selector.set_dist_impl(std::make_unique<precomputed_scores>(query.scores()));
        selector.initialize(args);
        for (unsigned distribution = 0; distribution < 3; ++distribution)
        {
            CAPTURE(distribution);
            if (distribution == 1)
                for (std::size_t i = 0; i < s.n; ++i)
                {
                    columns.price[i] = float(i % 7) - 3.0f;
                    columns.discount[i] = 0;
                }
            if (distribution == 2)
                std::fill(columns.price.begin(), columns.price.end(), 0.0f);
            const auto expected = applications::reference_topn(columns, s.k);
            query.upload(columns);
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                query.transform();
                selector.selection();
                query.output(selector.out_dist_gpu(), selector.out_label_gpu(),
                             !std::is_same_v<TestType, air_topk>);
                std::vector<applications::database_row> actual(s.k);
                query.download(actual);
                cuda_stream::make_default().sync();
                REQUIRE_NOTHROW(applications::verify_topn(columns, expected, actual));
                for (const auto& row : actual)
                    REQUIRE(row.row_id > std::numeric_limits<std::uint32_t>::max());
            }
        }
    }
}

TEST_CASE("GridSelect reinitializes workspace and returns sorted arbitrary-k results",
          "[applications][database]")
{
    grid_select selector;
    for (const auto [n, k] : {std::pair<std::size_t, std::size_t>{1, 1},
                              {37, 7},
                              {2053, 1025},
                              {600572, 2048},
                              {131, 33},
                              {1, 1}})
    {
        CAPTURE(n, k);
        const auto columns = table(n);
        const auto expected = applications::reference_topn(columns, k);
        applications::database_topn query(n, k);
        query.upload(columns);
        knn_args args{};
        args.point_count = n;
        args.query_count = 1;
        args.k = k;
        selector.set_dist_impl(std::make_unique<precomputed_scores>(query.scores()));
        selector.initialize(args);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            query.transform();
            selector.selection();
            query.output(selector.out_dist_gpu(), selector.out_label_gpu(), true);
            std::vector<applications::database_row> actual(k);
            query.download(actual);
            cuda_stream::make_default().sync();
            REQUIRE_NOTHROW(applications::verify_topn(columns, expected, actual));
        }
    }

    knn_args invalid{};
    invalid.point_count = std::size_t{std::numeric_limits<int>::max()} + 1;
    invalid.query_count = 1;
    invalid.k = 1;
    REQUIRE_THROWS_AS(selector.initialize(invalid), std::invalid_argument);
    std::swap(invalid.point_count, invalid.query_count);
    REQUIRE_THROWS_AS(selector.initialize(invalid), std::invalid_argument);
}

TEMPLATE_TEST_CASE("Database bits configurations handle partial blocks and maximum k",
                   "[applications][database]", bits_knn, bits_prefetch_knn, single_query_bits)
{
    const auto columns = table(2053);
    for (std::size_t k : {7u, 1025u, 2048u})
    {
        const auto expected = applications::reference_topn(columns, k);
        applications::database_topn query(columns.size(), k);
        query.upload(columns);
        for (std::size_t block : {128u, 256u, 512u})
            for (std::size_t batch : {1u, 4u, 7u, 8u, 13u, 16u})
            {
                CAPTURE(k, block, batch);
                TestType selector;
                knn_args args{};
                args.point_count = columns.size();
                args.query_count = 1;
                args.k = k;
                args.deg = 32;
                args.selection_block_size = block;
                args.items_per_thread = {batch, 1, 1};
                selector.set_dist_impl(std::make_unique<precomputed_scores>(query.scores()));
                selector.initialize(args);
                query.transform();
                selector.selection();
                query.output(selector.out_dist_gpu(), selector.out_label_gpu(), true);
                std::vector<applications::database_row> actual(k);
                query.download(actual);
                cuda_stream::make_default().sync();
                REQUIRE_NOTHROW(applications::verify_topn(columns, expected, actual));
            }
    }
}

TEST_CASE("Database output sorts arbitrary k and preserves value-label pairs",
          "[applications][database]")
{
    for (std::size_t k : {1u, 7u, 33u, 2048u})
    {
        CAPTURE(k);
        auto columns = table(2053);
        auto expected_rows = reference_rows(columns);
        expected_rows.resize(k);
        std::reverse(expected_rows.begin(), expected_rows.end());
        std::vector<float> scores;
        std::vector<std::int32_t> labels;
        for (const auto& row : expected_rows)
        {
            scores.push_back(-row.score);
            labels.push_back(row.source_index);
        }
        cuda_array<float, 2> selected_scores{{1, k}};
        cuda_array<std::int32_t, 2> selected_labels{{1, k}};
        applications::database_topn query(columns.size(), k);
        query.upload(columns);
        cuda_stream::make_default()
            .copy_to_gpu_async(selected_scores.view(), scores.data())
            .copy_to_gpu_async(selected_labels.view(), labels.data());
        query.output(selected_scores.view(), selected_labels.view(), false);
        std::vector<applications::database_row> actual(k);
        query.download(actual);
        cuda_stream::make_default().sync();
        REQUIRE_NOTHROW(
            applications::verify_topn(columns, applications::reference_topn(columns, k), actual));
    }
}

TEST_CASE("Database transform uses two correctly rounded FP32 operations",
          "[applications][database]")
{
    auto columns = table(4);
    columns.price = {0x1.000002p+0f, 0x1.fffffep+20f, 0x1p-126f, -0x1p-126f};
    columns.discount = {0x1p-24f, 0.13f, 0.5f, 0.5f};
    applications::database_topn query(4, 1);
    query.upload(columns);
    query.transform();
    std::vector<float> scores(4);
    cuda_stream::make_default().copy_from_gpu_async(scores.data(), query.scores()).sync();
    for (std::size_t i = 0; i < scores.size(); ++i)
        REQUIRE(scores[i] ==
                -applications::discounted_price(columns.price[i], columns.discount[i]));
}

TEST_CASE("Database validates raw values and projected output", "[applications][database-host]")
{
    const auto good = table(37);
    const auto expected = applications::reference_topn(good, 7);
    auto rows = reference_rows(good);
    rows.resize(7);
    REQUIRE_NOTHROW(applications::verify_topn(good, expected, rows));
    for (int error = 0; error < 6; ++error)
    {
        CAPTURE(error);
        auto wrong = rows;
        if (error == 0)
            wrong[0].payload += 1;
        if (error == 1)
            wrong[0].row_id += 1;
        if (error == 2)
            wrong[0].source_index = -1;
        if (error == 3)
            std::swap(wrong[0], wrong[6]);
        if (error == 4)
            wrong[0] = wrong[1];
        if (error == 5)
            wrong.pop_back();
        REQUIRE_THROWS(applications::verify_topn(good, expected, wrong));
    }
    auto equal = good;
    std::fill(equal.price.begin(), equal.price.end(), 0.0f);
    auto duplicate = reference_rows(equal);
    duplicate.resize(7);
    duplicate[0] = duplicate[1];
    REQUIRE_THROWS(
        applications::verify_topn(equal, applications::reference_topn(equal, 7), duplicate));
    for (float invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()})
        for (int column = 0; column < 3; ++column)
        {
            auto bad = good;
            (column == 0 ? bad.price : column == 1 ? bad.discount : bad.payload)[0] = invalid;
            REQUIRE_THROWS_AS(applications::validate_columns(bad), std::invalid_argument);
        }
    auto bad = good;
    bad.price[0] = std::numeric_limits<float>::max();
    bad.discount[0] = -1;
    REQUIRE_THROWS_AS(applications::validate_columns(bad), std::invalid_argument);
    bad = good;
    bad.payload.pop_back();
    REQUIRE_THROWS_AS(applications::validate_columns(bad), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::reference_topn(good, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::reference_topn(good, 38), std::invalid_argument);
    REQUIRE_THROWS_AS(applications::validate_columns({}), std::invalid_argument);
}

TEST_CASE("Database rejects shape errors and safely surfaces bad selector indices",
          "[applications][database]")
{
    REQUIRE_THROWS_AS((applications::database_topn{0, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::database_topn{2, 3}), std::invalid_argument);
    REQUIRE_THROWS_AS((applications::database_topn{4096, 2049}), std::invalid_argument);
    auto columns = table(3);
    applications::database_topn query(3, 3);
    REQUIRE_THROWS_AS(query.upload(table(2)), std::invalid_argument);
    REQUIRE_THROWS_AS(query.download({}), std::invalid_argument);
    REQUIRE_THROWS_AS(query.output({}, {}, false), std::invalid_argument);
    cuda_array<float, 2> scores{{1, 3}};
    cuda_array<std::int32_t, 2> indices{{1, 3}};
    const std::int32_t bad_indices[] = {-1, 3, std::numeric_limits<std::int32_t>::max()};
    const float values[] = {-3, -2, -1};
    query.upload(columns);
    cuda_stream::make_default()
        .copy_to_gpu_async(indices.view(), bad_indices)
        .copy_to_gpu_async(scores.view(), values);
    query.output(scores.view(), indices.view(), true);
    std::vector<applications::database_row> actual(3);
    query.download(actual);
    cuda_stream::make_default().sync();
    for (const auto& row : actual)
        REQUIRE(row.source_index == -1);
    REQUIRE_THROWS(
        applications::verify_topn(columns, applications::reference_topn(columns, 3), actual));
}
