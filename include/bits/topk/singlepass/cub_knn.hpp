#ifndef CUB_KNN_HPP_
#define CUB_KNN_HPP_

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

#endif // CUB_KNN_HPP_
