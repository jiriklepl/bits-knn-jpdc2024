#include <cstddef>

#include "bits/distance/eigen_distance.hpp"
#include "bits/knn_args.hpp"

void eigen_distance::prepare(const knn_args& args)
{
    args_ = args;

    points_ = Eigen::MatrixXf{args_.point_count, args_.dim};
    queries_ = Eigen::MatrixXf{args_.query_count, args_.dim};
    dist_ = Eigen::MatrixXf{args_.point_count, args_.query_count};

    for (std::size_t i = 0; i < args_.point_count; ++i)
    {
        for (std::size_t j = 0; j < args_.dim; ++j)
        {
            points_(i, j) = args.points[i * args_.dim + j];
        }
    }

    for (std::size_t i = 0; i < args_.query_count; ++i)
    {
        for (std::size_t j = 0; j < args_.dim; ++j)
        {
            queries_(i, j) = args.queries[i * args_.dim + j];
        }
    }

    for (std::size_t i = 0; i < args_.point_count; ++i)
    {
        for (std::size_t j = 0; j < args_.query_count; ++j)
        {
            dist_(i, j) = 0;
        }
    }
}

void eigen_distance::compute()
{
    dist_.noalias() = -2 * points_ * queries_.transpose();
    dist_.colwise() += points_.rowwise().squaredNorm();
}
