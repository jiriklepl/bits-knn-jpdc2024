#ifndef PARALLEL_KNN_HPP_
#define PARALLEL_KNN_HPP_

#include "bits/distance/eigen_distance.hpp"
#include "bits/knn.hpp"

/** Parallel CPU baseline kNN implementation.
 */
class parallel_knn : public knn
{
public:
    inline std::string id() const override { return "parallel"; }

    void initialize(const knn_args& args) override;
    void prepare() override;
    void selection() override;
    std::vector<distance_pair> finish() override;

private:
    std::vector<distance_pair> dist_;
};

#endif // PARALLEL_KNN_HPP_
