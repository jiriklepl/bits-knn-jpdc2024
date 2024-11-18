#ifndef BITS_DISTANCE_CUTLASS_DISTANCE_HPP_
#define BITS_DISTANCE_CUTLASS_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"
#include "bits/knn_args.hpp"

/** This class uses GEMM routines from cuBLAS to compute the distance
 */
class cutlass_distance : public cuda_distance
{
public:
    cutlass_distance() = default;

    void prepare(const knn_args& args) override;
    void compute() override;

    std::string name() const override { return "cutlass-dist"; }
};

#endif // BITS_DISTANCE_CUTLASS_DISTANCE_HPP_
