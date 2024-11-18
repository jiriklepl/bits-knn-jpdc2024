#ifndef BITS_TOPK_MULTIPASS_RADIK_KNN_HPP_
#define BITS_TOPK_MULTIPASS_RADIK_KNN_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "bits/cuda_knn.hpp"
#include "bits/knn_args.hpp"

class radik_knn : public cuda_knn
{
    static constexpr bool LARGEST = false;
    static constexpr bool ASCEND = true;
    static constexpr bool WITHSCALE = false;
    static constexpr bool WITHIDXIN = false;
    static constexpr bool PADDING = false;

public:
    std::string id() const override { return "radik"; }

    void initialize(const knn_args& args) override;
    void selection() override;

private:
    void* workspace_ = nullptr;
    std::size_t workspace_size_ = 0;
    std::vector<int> task_len_;
};

#endif // BITS_TOPK_MULTIPASS_RADIK_KNN_HPP_
