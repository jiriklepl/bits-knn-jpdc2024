#ifndef BITS_DISTANCE_MAGMA_DISTANCE_HPP_
#define BITS_DISTANCE_MAGMA_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"
#include "bits/knn_args.hpp"

/** Compute L2 distances using the modified matrix multiplication kernel from the MAGMA library.
 */
class magma_distance : public cuda_distance
{
public:
    void prepare(const knn_args& args) override;
    void compute() override;

    float transfer_seconds() const override
    {
        return transfer_end_.elapsed_seconds(transfer_begin_);
    }

    std::string name() const override { return "magma-dist"; }
};

/** Partial L2 distance using the modified matrix multiplication kernel from the MAGMA library.
 */
class magma_partial_distance : public magma_distance
{
public:
    void compute() override;

    float transfer_seconds() const override
    {
        return transfer_end_.elapsed_seconds(transfer_begin_);
    }

    std::string name() const override { return "magma-part-dist"; }
};

/** Compute KL divergence using the MAGMA library.
 */
class magma_kl_distance : public magma_distance
{
public:
    void compute() override;

    float transfer_seconds() const override
    {
        return transfer_end_.elapsed_seconds(transfer_begin_);
    }

    std::string name() const override { return "magma-kl-dist"; }
};

#endif // BITS_DISTANCE_MAGMA_DISTANCE_HPP_
