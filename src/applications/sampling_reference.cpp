#include "bits/applications/token_sampling.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace applications
{
void validate_logits(std::span<const float> logits, std::size_t batch, std::size_t vocabulary,
                     std::size_t k, float temperature)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (batch == 0 || batch > maximum || vocabulary == 0 || vocabulary > maximum ||
        batch > std::numeric_limits<std::size_t>::max() / vocabulary ||
        logits.size() != batch * vocabulary || k == 0 || k > vocabulary || k > 2048 ||
        !std::isfinite(temperature) || temperature <= 0)
        throw std::invalid_argument{"Invalid sampling shape, k, or temperature"};
    for (std::size_t row = 0; row < batch; ++row)
    {
        std::size_t finite = 0;
        for (std::size_t j = 0; j < vocabulary; ++j)
        {
            const float value = logits[row * vocabulary + j];
            if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
                throw std::invalid_argument{"Sampling logits contain NaN or positive infinity"};
            finite += std::isfinite(value);
        }
        if (finite < k)
            throw std::invalid_argument{"Each sampling row requires at least k finite logits"};
    }
}

std::vector<float> reference_sampling_topk(std::span<const float> logits, std::size_t batch,
                                           std::size_t vocabulary, std::size_t k)
{
    validate_logits(logits, batch, vocabulary, k, 1.0f);
    std::vector<float> expected(batch * k);
    std::vector<float> values(vocabulary);
    for (std::size_t row = 0; row < batch; ++row)
    {
        std::copy_n(logits.begin() + row * vocabulary, vocabulary, values.begin());
        if (k < vocabulary)
            std::nth_element(values.begin(), values.begin() + k, values.end(),
                             std::greater<float>{});
        std::sort(values.begin(), values.begin() + k, std::greater<float>{});
        std::copy_n(values.begin(), k, expected.begin() + row * k);
    }
    return expected;
}

void verify_sampling(std::span<const float> logits, std::size_t batch, std::size_t vocabulary,
                     std::size_t k, float temperature, std::span<const float> expected_topk,
                     std::span<const std::int32_t> candidate_ids,
                     std::span<const float> probabilities, std::span<const sampled_token> samples,
                     std::uint64_t seed, std::uint64_t draw)
{
    if (batch == 0 || vocabulary == 0 || k == 0 || k > vocabulary ||
        batch > std::numeric_limits<std::size_t>::max() / vocabulary ||
        logits.size() != batch * vocabulary || expected_topk.size() != batch * k ||
        candidate_ids.size() != batch * k || probabilities.size() != batch * k ||
        samples.size() != batch || !std::isfinite(temperature) || temperature <= 0)
        throw std::runtime_error{"Sampling verification shape or temperature is invalid"};
    std::vector<std::int32_t> unique(k);
    std::vector<float> values(k);
    std::vector<double> weights(k);
    for (std::size_t row = 0; row < batch; ++row)
    {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < k; ++j)
        {
            const auto id = candidate_ids[row * k + j];
            if (id < 0 || static_cast<std::size_t>(id) >= vocabulary)
                throw std::runtime_error{"Sampling returned an out-of-range candidate"};
            unique[j] = id;
            values[j] = logits[row * vocabulary + id];
            if (!std::isfinite(values[j]))
                throw std::runtime_error{"Sampling selected a masked or nonfinite logit"};
            maximum = std::max(maximum, static_cast<double>(values[j]));
        }
        std::sort(unique.begin(), unique.end());
        if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
            throw std::runtime_error{"Sampling returned duplicate candidates"};
        double total = 0;
        for (std::size_t j = 0; j < k; ++j)
        {
            weights[j] = std::exp((static_cast<double>(values[j]) - maximum) / temperature);
            total += weights[j];
        }
        std::sort(values.begin(), values.end(), std::greater<float>{});
        if (!std::equal(values.begin(), values.end(), expected_topk.begin() + row * k))
            throw std::runtime_error{"Sampling selected the wrong top-k logit multiset"};
        double probability_sum = 0;
        for (std::size_t j = 0; j < k; ++j)
        {
            const auto actual = probabilities[row * k + j];
            const auto expected = weights[j] / total;
            // Relative tolerance plus subnormal rounding covers FP32 output; it
            // still rejects a positive mass assigned to a zero-probability tail.
            const double tolerance =
                2e-6 * expected + 2.0 * std::numeric_limits<float>::denorm_min();
            if (!std::isfinite(actual) || actual < 0 || actual > 1 ||
                std::abs(static_cast<double>(actual) - expected) > tolerance)
                throw std::runtime_error{"Sampling softmax probabilities do not match logits"};
            probability_sum += actual;
        }
        if (std::abs(probability_sum - 1.0) > 2e-6)
            throw std::runtime_error{"Sampling probabilities are not normalized"};
        const double target = sampling_uniform(seed, draw, row) * probability_sum;
        double cumulative = 0;
        std::size_t chosen = 0;
        for (std::size_t j = 0; j < k; ++j)
        {
            if (probabilities[row * k + j] > 0)
                chosen = j;
            cumulative += probabilities[row * k + j];
            if (target < cumulative)
                break;
        }
        if (samples[row].token != candidate_ids[row * k + chosen] ||
            samples[row].probability != probabilities[row * k + chosen])
            throw std::runtime_error{"Sampling draw disagrees with its seeded candidate CDF"};
    }
}
} // namespace applications
