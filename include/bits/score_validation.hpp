#ifndef BITS_SCORE_VALIDATION_HPP_
#define BITS_SCORE_VALIDATION_HPP_

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

/** Validate host scores before upload, outside the GPU operator's timing boundary.
 * Positive infinity masks a candidate in the smallest-k representation. NaNs and
 * negative infinity are rejected. Each row must contain at least k finite scores.
 * Call again when replacing a trace; GPU transforms must preserve this contract.
 */
inline void validate_score_values(std::span<const float> scores, std::size_t rows,
                                  std::size_t columns, std::size_t k)
{
    if (rows == 0 || columns == 0 || k == 0 || k > columns ||
        rows > std::numeric_limits<std::size_t>::max() / columns || scores.size() != rows * columns)
    {
        throw std::invalid_argument{
            "Score dimensions must match the data and satisfy 0 < k <= columns"};
    }
    for (std::size_t row = 0; row < rows; ++row)
    {
        std::size_t finite_count = 0;
        for (float value : scores.subspan(row * columns, columns))
        {
            if (std::isfinite(value))
            {
                ++finite_count;
            }
            else if (value != std::numeric_limits<float>::infinity())
            {
                throw std::invalid_argument{"Scores must be finite or positive infinity (masked)"};
            }
        }
        if (finite_count < k)
        {
            throw std::invalid_argument{"Each score row must have at least k finite candidates"};
        }
    }
}

#endif // BITS_SCORE_VALIDATION_HPP_
