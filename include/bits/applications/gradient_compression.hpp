#ifndef BITS_APPLICATIONS_GRADIENT_COMPRESSION_HPP_
#define BITS_APPLICATIONS_GRADIENT_COMPRESSION_HPP_

#include "bits/cuda_array.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace applications
{
struct gradient_entry
{
    std::int32_t index;
    float value;
};

void validate_gradient(std::span<const float> gradient);
// Descending magnitudes; ties permit any valid subset of indices in the output.
std::vector<float> reference_gradient_topk(std::span<const float> gradient, std::size_t k);
void verify_gradient_topk(std::span<const float> gradient,
                          std::span<const float> expected_magnitudes,
                          std::span<const gradient_entry> actual);

/** Stateless global top-k compression of one complete FP32 gradient tensor. */
class gradient_compression
{
public:
    gradient_compression(std::size_t elements, std::size_t k);
    // Validate values with validate_gradient() before upload; host data must outlive the copy.
    void upload(std::span<const float> gradient);
    void transform();
    // Output order follows the selector. Values are gathered from the original tensor.
    void output(array_view<float, 2> selected_scores, array_view<std::int32_t, 2> selected_indices);
    void download(std::span<gradient_entry> destination) const;

    array_view<float, 2> scores() const { return scores_.view(); }

private:
    std::size_t elements_, k_;
    cuda_array<float, 1> gradient_;
    cuda_array<float, 2> scores_;
    cuda_array<gradient_entry, 1> result_;
};
} // namespace applications
#endif
