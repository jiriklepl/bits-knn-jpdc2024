#ifndef BITS_TOPK_SINGLEPASS_FUSED_KNN_HPP_
#define BITS_TOPK_SINGLEPASS_FUSED_KNN_HPP_

#include <string>

#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/knn_args.hpp"

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

#endif // BITS_TOPK_SINGLEPASS_FUSED_KNN_HPP_
