#ifndef BITS_TOPK_SINGLEPASS_CUB_KNN_HPP_
#define BITS_TOPK_SINGLEPASS_CUB_KNN_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class cub_knn : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "buffered-cub"; }
};

class cub_direct : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "buffered-cub-direct"; }
};

#endif // BITS_TOPK_SINGLEPASS_CUB_KNN_HPP_
