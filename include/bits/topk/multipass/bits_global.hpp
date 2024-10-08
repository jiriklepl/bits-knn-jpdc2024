#ifndef BITS_GLOBAL_HPP_
#define BITS_GLOBAL_HPP_

#include <string>
#include <vector>

#include "bits/cuda_knn.hpp"

// buffered k-selection using bitonic sort for larger k
class bits_global : public cuda_knn
{
public:
    void selection() override;

    std::string id() const override { return "bits-global"; }

    std::vector<knn::pair_t> finish() override;
};

#endif // BITS_GLOBAL_HPP_
