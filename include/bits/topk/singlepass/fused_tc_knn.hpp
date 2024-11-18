#ifndef BITS_TOPK_SINGLEPASS_FUSED_TC_KNN_HPP_
#define BITS_TOPK_SINGLEPASS_FUSED_TC_KNN_HPP_

#include <string>

#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/knn_args.hpp"
#include "bits/topk/singlepass/fused_tc_policy.hpp"

template <typename Policy>
class fused_tc_knn : public cuda_knn
{
public:
    using input_t = typename Policy::input_t;
    using output_t = typename Policy::output_t;

    static constexpr auto QUERY_TILE_SIZE = Policy::QUERY_TILE_SIZE;
    static constexpr auto POINT_TILE_SIZE = Policy::POINT_TILE_SIZE;
    static constexpr auto DIM_TILE_SIZE = Policy::DIM_TILE_SIZE;

    void initialize(const knn_args& args) override;

    void distances() override {}

    void selection() override;

protected:
    cuda_array<float, 2> points_gpu_;
    cuda_array<float, 2> queries_gpu_;

    cuda_array<input_t, 2> in_points_gpu_;
    cuda_array<float, 1> in_point_norms_gpu_;

    cuda_array<input_t, 2> in_queries_gpu_;
    cuda_array<float, 1> in_query_norms_gpu_;
};

class fused_tc_half_knn : public fused_tc_knn<fused_tc_half_policy>
{
    std::string id() const override { return "fused-tc-half"; }
};

class fused_tc_bfloat16_knn : public fused_tc_knn<fused_tc_bfloat16_policy>
{
    std::string id() const override { return "fused-tc-bfloat16"; }
};

class fused_tc_double_knn : public fused_tc_knn<fused_tc_double_policy>
{
    std::string id() const override { return "fused-tc-double"; }
};

#endif // BITS_TOPK_SINGLEPASS_FUSED_TC_KNN_HPP_
