#ifndef BITS_TOPK_SINGLEPASS_PARTIAL_BITONIC_BUFFERED_HPP_
#define BITS_TOPK_SINGLEPASS_PARTIAL_BITONIC_BUFFERED_HPP_

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

#endif // BITS_TOPK_SINGLEPASS_PARTIAL_BITONIC_BUFFERED_HPP_
