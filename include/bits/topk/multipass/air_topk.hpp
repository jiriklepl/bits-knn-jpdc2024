#ifndef BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_
#define BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class air_topk : public cuda_knn
{
public:
    std::string id() const override { return "air-topk"; }

    void selection() override;
};

#endif // BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_
