#ifndef EIGEN_DISTANCE_HPP_
#define EIGEN_DISTANCE_HPP_

#include <string>

#include <Eigen/Dense>

#include "bits/array_view.hpp"
#include "bits/distance/cuda_distance.hpp"

class eigen_distance : public cuda_distance
{
public:
    std::string name() const override { return "eigen-dist"; }

    void prepare(const knn_args& args) override;
    void compute() override;

    array_view<float, 2> matrix_gpu() const override { return array_view<float, 2>{}; }

    inline array_view<float, 2> matrix_cpu() override
    {
        return {dist_.data(), {args_.query_count, args_.point_count}, {args_.point_count, 1}};
    }

private:
    Eigen::MatrixXf points_;
    Eigen::MatrixXf queries_;
    Eigen::MatrixXf dist_;
};

#endif // EIGEN_DISTANCE_HPP_
