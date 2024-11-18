#ifndef BITS_TOPK_PARALLEL_KNN_HPP_
#define BITS_TOPK_PARALLEL_KNN_HPP_

#include <string>
#include <vector>

#include "bits/knn.hpp"
#include "bits/knn_args.hpp"

/** Parallel CPU baseline kNN implementation.
 */
class parallel_knn : public knn
{
public:
    std::string id() const override { return "parallel"; }

    void initialize(const knn_args& args) override;
    void prepare() override;
    void selection() override;
    std::vector<distance_pair> finish() override;

private:
    std::vector<distance_pair> dist_;
};

#endif // BITS_TOPK_PARALLEL_KNN_HPP_
