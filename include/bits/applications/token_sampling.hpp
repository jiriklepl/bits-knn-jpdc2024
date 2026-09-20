#ifndef BITS_APPLICATIONS_TOKEN_SAMPLING_HPP_
#define BITS_APPLICATIONS_TOKEN_SAMPLING_HPP_

#include "bits/cuda_array.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace applications
{
struct sampled_token
{
    std::int32_t token;
    float probability;
};

// Stateless SplitMix64 mixing. A draw is indexed by (seed, draw, sequence), so repeated
// invocations need no mutable generator state. This is not a cryptographic generator.
__host__ __device__ inline std::uint64_t sampling_mix(std::uint64_t value)
{
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Exactly representable 53-bit uniform variate in [0, 1), shared by CPU verification.
__host__ __device__ inline double sampling_uniform(std::uint64_t seed, std::uint64_t draw,
                                                   std::uint64_t sequence)
{
    const auto key =
        sampling_mix(seed + 0x9e3779b97f4a7c15ULL) ^ sampling_mix(draw + 0xd1b54a32d192ed03ULL);
    return static_cast<double>(sampling_mix(key + sequence * 0x9e3779b97f4a7c15ULL) >> 11) *
           0x1p-53;
}

// Contiguous batch x vocabulary FP32 logits; -infinity masks are allowed, with at
// least k finite logits per row. NaNs and +infinity are rejected.
void validate_logits(std::span<const float> logits, std::size_t batch, std::size_t vocabulary,
                     std::size_t k, float temperature);
std::vector<float> reference_sampling_topk(std::span<const float> logits, std::size_t batch,
                                           std::size_t vocabulary, std::size_t k);
void verify_sampling(std::span<const float> logits, std::size_t batch, std::size_t vocabulary,
                     std::size_t k, float temperature, std::span<const float> expected_topk,
                     std::span<const std::int32_t> candidate_ids,
                     std::span<const float> probabilities, std::span<const sampled_token> samples,
                     std::uint64_t seed, std::uint64_t draw);

/** GPU top-k sampling operator. Allocation and upload are outside repeated execution. */
class token_sampling
{
public:
    token_sampling(std::size_t batch, std::size_t vocabulary, std::size_t k, float temperature);
    // Validate host values with validate_logits() before upload; they must outlive the copy.
    void upload(std::span<const float> logits);
    void transform();
    // Retains candidate order: a selector permutation may change the seeded sample.
    // Output owns a copy of the candidate IDs and FP32 probabilities for verification.
    void output(array_view<float, 2> selected_scores, array_view<std::int32_t, 2> selected_indices,
                std::uint64_t seed, std::uint64_t draw);
    void download(std::span<sampled_token> destination) const;
    void download_distribution(std::span<std::int32_t> candidate_ids,
                               std::span<float> probabilities) const;

    array_view<float, 2> scores() const { return scores_.view(); }

private:
    std::size_t batch_, vocabulary_, k_;
    float temperature_;
    cuda_array<float, 2> logits_, scores_, probabilities_;
    cuda_array<std::int32_t, 2> candidate_ids_;
    cuda_array<sampled_token, 1> result_;
};
} // namespace applications
#endif
