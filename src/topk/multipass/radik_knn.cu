#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <radik/RadixSelect/topk_radixselect.h>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/knn_args.hpp"
#include "bits/topk/multipass/radik_knn.hpp"

void radik_knn::initialize(const knn_args& args)
{
    cuda_knn::initialize(args);

    task_len_ = std::vector<int>(query_count(), point_count());

    const auto max_task_len = *std::max_element(task_len_.begin(), task_len_.end());

    // retrieve the workspace size in bytes
    getRadixSelectLWorkSpaceSize<float>(k(), max_task_len, query_count(), &workspace_size_);

    CUCH(cudaMalloc(&workspace_, workspace_size_));
}

void radik_knn::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    auto stream = cuda_stream::make_default();

    topKRadixSelectL<std::int32_t, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, float, PADDING>(
        in_dist.data(), nullptr, out_dist.data(), out_label.data(), workspace_, task_len_.data(),
        query_count(), k(), stream.get());

    cuda_stream::make_default().sync();
}
