#ifndef BITS_TOPK_SINGLEPASS_FUSED_CACHE_KNN_
#define BITS_TOPK_SINGLEPASS_FUSED_CACHE_KNN_

#include <string>

#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/knn_args.hpp"

class fused_cache_knn : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;
    void distances() override;
    void selection() override;

    std::string id() const override { return "fused-cache"; }

protected:
    cuda_array<float, 2> points_gpu_;
    cuda_array<float, 2> queries_gpu_;
};

#endif // BITS_TOPK_SINGLEPASS_FUSED_CACHE_KNN_
