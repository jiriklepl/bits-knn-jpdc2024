#ifndef BASELINE_DISTANCE_HPP_
#define BASELINE_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"

/** Baseline CUDA kernel for computing distances
 */
class baseline_distance : public cuda_distance
{
public:
    void prepare(const knn_args& args) override;
    void compute() override;

    inline std::string name() const override { return "baseline-dist"; }
};

#endif // BASELINE_DISTANCE_HPP_
