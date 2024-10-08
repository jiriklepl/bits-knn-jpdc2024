#ifndef RAPIDSAI_FUSED_HPP_
#define RAPIDSAI_FUSED_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class rapidsai_fused : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;
    void distances() override;
    void selection() override;

    std::string id() const override { return "rapidsai-fused"; }

protected:
    cuda_array<float, 2> points_gpu_;
    cuda_array<float, 2> queries_gpu_;

    bool row_major_index_ = true;
    bool row_major_query_ = true;
};

#endif // RAPIDSAI_FUSED_HPP_
