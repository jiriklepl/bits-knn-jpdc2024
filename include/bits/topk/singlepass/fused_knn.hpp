#ifndef FUSED_KNN_
#define FUSED_KNN_

#include <string>

#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"

class fused_regs_knn : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;
    void distances() override;
    void selection() override;

    std::string id() const override { return "fused-regs"; }

protected:
    cuda_array<float, 2> points_gpu_;
    cuda_array<float, 2> queries_gpu_;
};

// fused kernel which takes block size and items per thread parameters from
class fused_regs_knn_tunable : public fused_regs_knn
{
public:
    void selection() override;

    std::string id() const override { return "fused-regs-tunable"; }
};

#endif // FUSED_KNN_
