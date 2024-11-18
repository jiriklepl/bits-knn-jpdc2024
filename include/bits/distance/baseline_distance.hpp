#ifndef BITS_DISTANCE_BASELINE_DISTANCE_HPP_
#define BITS_DISTANCE_BASELINE_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"
#include "bits/knn_args.hpp"

/** Baseline CUDA kernel for computing distances
 */
class baseline_distance : public cuda_distance
{
public:
    void prepare(const knn_args& args) override;
    void compute() override;

    std::string name() const override { return "baseline-dist"; }
};

#endif // BITS_DISTANCE_BASELINE_DISTANCE_HPP_
