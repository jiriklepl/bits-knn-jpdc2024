#include "bits/applications/gradient_compression.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace applications
{
void validate_gradient(std::span<const float> gradient)
{
    if (gradient.empty() ||
        gradient.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument{"Gradient must have a positive length within int32 indexing"};
    for (float value : gradient)
        if (!std::isfinite(value))
            throw std::invalid_argument{"Gradient values must be finite"};
}

std::vector<float> reference_gradient_topk(std::span<const float> gradient, std::size_t k)
{
    validate_gradient(gradient);
    if (k == 0 || k > gradient.size())
        throw std::invalid_argument{"Require 0 < k <= gradient elements"};
    std::vector<float> magnitudes(gradient.size());
    std::transform(gradient.begin(), gradient.end(), magnitudes.begin(),
                   [](float value) { return std::fabs(value); });
    if (k < magnitudes.size())
        std::nth_element(magnitudes.begin(), magnitudes.begin() + k, magnitudes.end(),
                         std::greater<float>{});
    magnitudes.resize(k);
    std::sort(magnitudes.begin(), magnitudes.end(), std::greater<float>{});
    return magnitudes;
}

void verify_gradient_topk(std::span<const float> gradient, std::span<const float> expected,
                          std::span<const gradient_entry> actual)
{
    if (actual.empty() || actual.size() != expected.size() || actual.size() > gradient.size())
        throw std::runtime_error{"Gradient top-k output has the wrong length"};
    std::vector<std::int32_t> indices;
    std::vector<float> magnitudes;
    indices.reserve(actual.size());
    magnitudes.reserve(actual.size());
    for (const auto& entry : actual)
    {
        if (entry.index < 0 || static_cast<std::size_t>(entry.index) >= gradient.size())
            throw std::runtime_error{"Gradient top-k returned an invalid source index"};
        // A bitwise comparison also preserves the sign of zero in the original tensor.
        if (!std::isfinite(entry.value) || std::bit_cast<std::uint32_t>(entry.value) !=
                                               std::bit_cast<std::uint32_t>(gradient[entry.index]))
            throw std::runtime_error{"Gradient top-k did not preserve the original signed value"};
        indices.push_back(entry.index);
        magnitudes.push_back(std::fabs(entry.value));
    }
    std::sort(indices.begin(), indices.end());
    if (std::adjacent_find(indices.begin(), indices.end()) != indices.end())
        throw std::runtime_error{"Gradient top-k returned duplicate source indices"};
    std::sort(magnitudes.begin(), magnitudes.end(), std::greater<float>{});
    if (!std::equal(magnitudes.begin(), magnitudes.end(), expected.begin()))
        throw std::runtime_error{"Gradient top-k magnitude verification failed"};
}
} // namespace applications
