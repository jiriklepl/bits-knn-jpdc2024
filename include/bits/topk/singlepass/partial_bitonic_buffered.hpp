#ifndef PARTIAL_BITONIC_BUFFERED_HPP_
#define PARTIAL_BITONIC_BUFFERED_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class buffered_partial_bitonic : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-buffered"; }
};

class static_buffered_partial_bitonic : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-buffered-static"; }
};

#endif // PARTIAL_BITONIC_BUFFERED_HPP_
