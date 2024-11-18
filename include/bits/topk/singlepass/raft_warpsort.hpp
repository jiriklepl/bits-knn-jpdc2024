#ifndef BITS_TOPK_MULTIPASS_RAFT_WARPSORT_HPP_
#define BITS_TOPK_MULTIPASS_RAFT_WARPSORT_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

class raft_warpsort : public cuda_knn
{
public:
    std::string id() const override { return "warpsort"; }

    void selection() override;
};

#endif // BITS_TOPK_MULTIPASS_RAFT_WARPSORT_HPP_
