#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "bits/verify.hpp"

#include "bits/knn.hpp"

namespace
{

// adaptation of float comparison implementation from
// https://floating-point-gui.de/errors/comparison/ for C++
template <typename Float>
bool approx_equal(Float a, Float b, Float epsilon)
{
    const auto abs_a = std::abs(a);
    const auto abs_b = std::abs(b);
    const auto diff = std::abs(a - b);

    if (a == b)
    {
        return true;
    }

    if (a == 0 || b == 0 || (abs_a + abs_b < std::numeric_limits<Float>::min()))
    {
        return diff < (epsilon * std::numeric_limits<Float>::min());
    }

    return diff / std::min<Float>(abs_a + abs_b, std::numeric_limits<Float>::max()) < epsilon;
}

void normalize(std::vector<knn::pair_t>& data)
{
    const float EPSILON = 1e-3;

    for (std::int32_t i = 1; i < (std::int32_t)data.size(); ++i)
    {
        auto j = i;
        while (j > 0 && approx_equal(data[j].distance, data[j - 1].distance, EPSILON) &&
               data[j].index > data[j - 1].index)
        {
            std::swap(data[j], data[j - 1]);
            --j;
        }
    }
}

// returns true iff all values in the [begin, end) range have the same distance
template <typename It>
bool is_distance_same(It begin, It end)
{
    const float EPSILON = 1e-3;

    if (begin == end)
    {
        return false;
    }

    auto prev = begin;
    for (auto it = begin + 1; it != end; ++it)
    {
        if (!approx_equal(prev->distance, it->distance, EPSILON))
        {
            return false;
        }
        prev = it;
    }
    return true;
}

} // namespace

bool verify(const std::vector<knn::pair_t>& expected_result,
            const std::vector<knn::pair_t>& actual_result, std::size_t k)
{
    const float EPSILON = 1e-3;

    auto expected = expected_result;
    auto actual = actual_result;

    normalize(expected);
    normalize(actual);

    bool result = true;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto expected_label = expected[i].index;
        const auto actual_label = actual[i].index;
        if (actual_label != expected_label)
        {
            const auto last = i + (k - (i % k));
            if (is_distance_same(expected.begin() + i, expected.begin() + last) &&
                is_distance_same(actual.begin() + i, actual.begin() + last) &&
                approx_equal(expected[i].distance, actual[i].distance, EPSILON))
            {
                continue;
            }

            std::cerr << "Difference at index " << i << " (expected: " << expected_label
                      << ", actual: " << actual_label << ")" << '\n';
            std::cerr << "- expected distance: " << expected[i].distance << '\n';
            std::cerr << "- actual distance: " << actual[i].distance << '\n';
            result = false;
        }
    }
    return result;
}
