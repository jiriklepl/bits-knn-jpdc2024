#include "bits/applications/database_topn.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace applications
{
float discounted_price(float price, float discount)
{
    // The volatile intermediate prevents contraction into a differently rounded expression.
    volatile float factor = 1.0f - discount;
    return price * factor;
}

void validate_columns(const database_columns& columns)
{
    const auto n = columns.size();
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        columns.discount.size() != n || columns.payload.size() != n || columns.row_id.size() != n)
        throw std::invalid_argument{
            "Database columns must have the same positive length within int32 indexing"};
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(columns.price[i]) || !std::isfinite(columns.discount[i]) ||
            !std::isfinite(columns.payload[i]) ||
            !std::isfinite(discounted_price(columns.price[i], columns.discount[i])))
            throw std::invalid_argument{"Database values and FP32 ranking scores must be finite"};
    }
}

std::vector<float> reference_topn(const database_columns& columns, std::size_t k)
{
    validate_columns(columns);
    if (k == 0 || k > columns.size())
        throw std::invalid_argument{"Require 0 < k <= rows"};
    std::vector<float> scores(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i)
        scores[i] = discounted_price(columns.price[i], columns.discount[i]);
    if (k < scores.size())
        std::nth_element(scores.begin(), scores.begin() + k, scores.end(), std::greater<float>{});
    scores.resize(k);
    std::sort(scores.begin(), scores.end(), std::greater<float>{});
    return scores;
}

void verify_topn(const database_columns& columns, std::span<const float> expected,
                 std::span<const database_row> actual)
{
    if (actual.size() != expected.size())
        throw std::runtime_error{"Top-N output has the wrong length"};
    std::vector<std::int32_t> indices;
    indices.reserve(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const auto& row = actual[i];
        if (row.source_index < 0 || static_cast<std::size_t>(row.source_index) >= columns.size())
            throw std::runtime_error{"Top-N returned an invalid source index"};
        const auto j = static_cast<std::size_t>(row.source_index);
        if (row.score != expected[i] ||
            row.score != discounted_price(columns.price[j], columns.discount[j]) ||
            row.row_id != columns.row_id[j] || row.payload != columns.payload[j])
            throw std::runtime_error{
                "Top-N score, ordering, row ID, or payload verification failed"};
        indices.push_back(row.source_index);
    }
    std::sort(indices.begin(), indices.end());
    if (std::adjacent_find(indices.begin(), indices.end()) != indices.end())
        throw std::runtime_error{"Top-N returned duplicate source indices"};
}
} // namespace applications
