#ifndef BITS_TOPK_SINGLEPASS_WARP_SELECT_HPP_
#define BITS_TOPK_SINGLEPASS_WARP_SELECT_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class warp_select : public cuda_knn
{
public:
    std::string id() const override { return "warp-select"; }

    void selection() override;
};

class block_select : public warp_select
{
public:
    std::string id() const override { return "block-select"; }

    void selection() override;
};

class block_select_tunable : public warp_select
{
public:
    std::string id() const override { return "block-select-tunable"; }

    void selection() override;
};

class warp_select_tunable : public warp_select
{
public:
    std::string id() const override { return "warp-select-tunable"; }

    void selection() override;
};

#endif // BITS_TOPK_SINGLEPASS_WARP_SELECT_HPP_
