#ifndef BITS_DISTANCE_DIST_RUNNER_HPP_
#define BITS_DISTANCE_DIST_RUNNER_HPP_

#include <string>

#include "bits/cuda_knn.hpp"

template <typename Distance>
class dist_runner : public cuda_knn
{
public:
    dist_runner() { set_dist_impl(std::make_unique<Distance>()); }

    std::string id() const override { return dist_impl_->name(); }
};

#endif // BITS_DISTANCE_DIST_RUNNER_HPP_
