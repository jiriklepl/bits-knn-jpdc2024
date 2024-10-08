#ifndef PARTIAL_BITONIC_HPP_
#define PARTIAL_BITONIC_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class partial_bitonic : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic"; }
};

class partial_bitonic_warp : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-warp"; }
};

class partial_bitonic_warp_static : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-warp-static"; }
};

class partial_bitonic_arrays : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-soa"; }
};

class partial_bitonic_regs : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "partial-bitonic-regs"; }
};

#endif // PARTIAL_BITONIC_HPP_
