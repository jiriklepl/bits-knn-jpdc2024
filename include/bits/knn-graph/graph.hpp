#ifndef KNN_GRAPH_HPP_
#define KNN_GRAPH_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bits/cuda_array.hpp"
#include "bits/knn_args.hpp"

/** kNN graph construction
 */
class knn_graph
{
public:
    inline static constexpr std::size_t CUDA_STREAMS = 2;

    void prepare(const knn_args& args);
    void run();

protected:
    knn_args args_;
    cuda_array<float, 2> db_gpu_;
    std::array<cuda_array<float, 2>, CUDA_STREAMS> dist_gpu_;
    std::array<cuda_array<float, 2>, CUDA_STREAMS> topk_dist_gpu_;
    std::array<cuda_array<std::int32_t, 2>, CUDA_STREAMS> topk_label_gpu_;
    std::vector<std::int32_t> topk_label_;
};

#endif // KNN_GRAPH_HPP_
