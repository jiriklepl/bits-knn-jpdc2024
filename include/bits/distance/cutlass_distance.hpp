#ifndef CUTLASS_DISTANCE_HPP_
#define CUTLASS_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"

/** This class uses GEMM routines from cuBLAS to compute the distance
 */
class cutlass_distance : public cuda_distance
{
public:
    cutlass_distance() = default;

    void prepare(const knn_args& args) override;
    void compute() override;

    inline std::string name() const override { return "cutlass-dist"; }
};

#endif // CUTLASS_DISTANCE_HPP_
