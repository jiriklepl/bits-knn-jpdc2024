#ifndef BITS_APPLICATIONS_DATABASE_TOPN_HPP_
#define BITS_APPLICATIONS_DATABASE_TOPN_HPP_

#include "bits/cuda_array.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace applications
{
struct database_columns
{
    std::vector<float> price, discount, payload;
    std::vector<std::uint64_t> row_id;

    std::size_t size() const { return price.size(); }
};

struct database_row
{
    std::uint64_t row_id;
    float score;
    float payload;
    std::int32_t source_index;
};

// Two FP32 operations, with rounding after subtraction and multiplication.
float discounted_price(float price, float discount);
void validate_columns(const database_columns& columns);
std::vector<float> reference_topn(const database_columns& columns, std::size_t k);
void verify_topn(const database_columns& columns, std::span<const float> expected_scores,
                 std::span<const database_row> actual);

/** GPU-resident top-N operator; allocation and upload are separate from repeated execution. */
class database_topn
{
public:
    database_topn(std::size_t rows, std::size_t k);
    // Validate values with validate_columns() before upload; host data must outlive the copy.
    void upload(const database_columns& columns);
    void transform();
    // Sort the selected buffers in place only when the backend does not guarantee order.
    void output(array_view<float, 2> selected_scores, array_view<std::int32_t, 2> selected_indices,
                bool already_sorted);
    void download(std::span<database_row> destination) const;

    array_view<float, 2> scores() const { return scores_.view(); }

private:
    std::size_t rows_, k_;
    cuda_array<float, 1> price_, discount_, payload_;
    cuda_array<std::uint64_t, 1> row_id_;
    cuda_array<float, 2> scores_;
    cuda_array<database_row, 1> result_;
};
} // namespace applications
#endif
