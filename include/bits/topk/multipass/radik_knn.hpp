#ifndef RADIK_KNN_HPP_
#define RADIK_KNN_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "bits/cuda_knn.hpp"
#include "bits/knn_args.hpp"

class radik_knn : public cuda_knn
{
    static constexpr bool LARGEST = 0;
    static constexpr bool ASCEND = 1;
    static constexpr bool WITHSCALE = 0;
    static constexpr bool WITHIDXIN = 0;
    static constexpr bool PADDING = 0;

public:
    std::string id() const override { return "radik"; }

    void initialize(const knn_args& args) override;
    void selection() override;

private:
    void* workspace_ = nullptr;
    std::size_t workspace_size_ = 0;
    std::vector<int> task_len_ = {};
};

#endif // RADIK_KNN_HPP_
