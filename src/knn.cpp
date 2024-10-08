#include <memory>
#include <utility>

#include "bits/cuda_stream.hpp"
#include "bits/knn.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"

#include "bits/distance/magma_distance.hpp"

void knn::initialize(const knn_args& args)
{
    args_ = args;
    // enforce row major layout, implementations can change this
    args_.dist_layout = matrix_layout::row_major;
    // use the magma approach by default
    if (dist_impl_ == nullptr)
    {
        dist_impl_ = std::make_unique<magma_distance>();
    }
    dist_impl_->prepare(args_);
}

void knn::distances()
{
    dist_impl_->compute();

    cuda_stream::make_default().sync();
}

void knn::set_dist_impl(std::unique_ptr<cuda_distance>&& impl) { dist_impl_ = std::move(impl); }
