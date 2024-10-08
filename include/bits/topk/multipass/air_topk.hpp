#ifndef AIR_TOPK_HPP_
#define AIR_TOPK_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class air_topk : public cuda_knn
{
public:
    std::string id() const override { return "air-topk"; }

    void selection() override;
};

#endif // AIR_TOPK_HPP_
