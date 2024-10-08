#ifndef BITS_KNN_HPP_
#define BITS_KNN_HPP_

#include <cstdint>
#include <string>

#include "bits/array_view.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/distance/baseline_distance.hpp"

/** Multi-query, single-pass kernel based on Bitonic sort.
 */
class bits_knn : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;
    void selection() override;

    std::string id() const override { return "bits"; }
};

/** The same as `bits_knn` with additional prefetch instructions.
 */
class bits_prefetch_knn : public bits_knn
{
public:
    void selection() override;

    std::string id() const override { return "bits-prefetch"; }
};

/** Single-query adaptation of `bits_knn`.
 */
class single_query_bits : public cuda_knn
{
public:
    void initialize(const knn_args& args) override;
    void selection() override;

    std::string id() const override { return "bits-sq"; }

private:
    // partial top k results
    cuda_array<float, 2> tmp_dist_;
    cuda_array<std::int32_t, 2> tmp_label_;
    cuda_array<std::int32_t, 1> label_offsets_;
};

#endif // BITS_KNN_HPP_
