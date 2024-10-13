#ifndef GRID_SELECT_HPP_
#define GRID_SELECT_HPP_

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "bits/cuch.hpp"
#include "bits/cuda_knn.hpp"

class grid_select : public cuda_knn
{
public:
    ~grid_select() override
    {
        if (buf_)
        {
            CUCH(cudaFree(buf_));
            buf_ = nullptr;
        }
    }

    std::string id() const override { return "grid-select"; }

    void selection() override;

private:
    std::size_t buf_size_ = 0;
    void* buf_ = nullptr;
};

#endif // GRID_SELECT_HPP_
