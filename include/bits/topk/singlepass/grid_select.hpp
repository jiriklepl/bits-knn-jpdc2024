#ifndef BITS_TOPK_SINGLEPASS_GRID_SELECT_HPP_
#define BITS_TOPK_SINGLEPASS_GRID_SELECT_HPP_

#include <cstddef>
#include <string>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_ptr.hpp"

class grid_select : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;

    std::string id() const override { return "grid-select"; }

    void selection() override;

private:
    std::size_t buf_size_ = 0;
    cuda_ptr<std::byte> buf_;
};

#endif // BITS_TOPK_SINGLEPASS_GRID_SELECT_HPP_
