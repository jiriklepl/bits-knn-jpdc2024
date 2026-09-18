#ifndef BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_
#define BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_

#include <memory>
#include <string>

#include "bits/cuda_knn.hpp"

namespace raft
{
class resources;
}

class air_topk : public cuda_knn
{
public:
    air_topk();
    ~air_topk() override;

    void initialize(const knn_args& args) override;

    std::string id() const override { return "air-topk"; }

    void selection() override;

private:
    std::unique_ptr<raft::resources> resources_;
};

#endif // BITS_TOPK_MULTIPASS_AIR_TOPK_HPP_
