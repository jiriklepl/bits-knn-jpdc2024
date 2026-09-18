#ifndef BITS_DISTANCE_PRECOMPUTED_SCORES_HPP_
#define BITS_DISTANCE_PRECOMPUTED_SCORES_HPP_

#include <limits>
#include <stdexcept>
#include <string>

#include "bits/distance/cuda_distance.hpp"

/** Non-owning, contiguous FP32 score matrix for the existing selection wrappers.
 * Rows are independent selections; columns are candidates. Smaller scores win.
 * The caller owns the device storage and must keep it alive and at a fixed address
 * through every selection. Values may be updated between calls on the default
 * stream (or synchronized with it). This provider does not transform or validate
 * score values: validate host data with validate_score_values() before upload.
 * Callers must exclude NaNs and provide at least k finite values
 * per row when using positive infinity to mask candidates.
 */
class precomputed_scores : public cuda_distance
{
public:
    explicit precomputed_scores(array_view<float, 2> scores) : scores_(scores)
    {
        if (scores.data() == nullptr || scores.size(0) == 0 || scores.size(1) == 0 ||
            scores.stride(0) != scores.size(1) || scores.stride(1) != 1 ||
            scores.size(0) >
                std::numeric_limits<std::size_t>::max() / sizeof(float) / scores.size(1))
        {
            throw std::invalid_argument{
                "Precomputed scores require a nonempty contiguous FP32 device matrix"};
        }
    }

    void prepare(const knn_args& args) override
    {
        if (args.query_count != scores_.size(0) || args.point_count != scores_.size(1) ||
            args.dist_layout != matrix_layout::row_major)
        {
            throw std::invalid_argument{
                "Precomputed score shape/layout does not match the selection arguments"};
        }
        // Deliberately avoid cuda_distance::prepare(): there are no point/query matrices to
        // allocate.
        args_ = args;
    }

    void compute() override {}

    array_view<float, 2> matrix_gpu() const override { return scores_; }

    std::string name() const override { return "precomputed-scores"; }

private:
    array_view<float, 2> scores_;
};

#endif // BITS_DISTANCE_PRECOMPUTED_SCORES_HPP_
