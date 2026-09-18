#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_radix.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/multipass/air_topk.hpp"

air_topk::air_topk() = default;
air_topk::~air_topk() = default;

void air_topk::initialize(const knn_args& args)
{
    if (args.k == 0 || args.k > args.point_count || args.query_count == 0 ||
        args.point_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        args.query_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument{"AIR Top-K requires 0 < k <= point_count and positive "
                                    "point/query counts representable by 32-bit indices"};
    }

    resources_.reset();
    cuda_knn::initialize(args);
    resources_ = std::make_unique<raft::resources>();
    raft::resource::set_cuda_stream(*resources_,
                                    rmm::cuda_stream_view{cuda_stream::make_default().get()});
    // Reuse workspace across selections without replacing the process-wide RMM allocator.
    raft::resource::set_workspace_to_pool_resource(*resources_);
}

void air_topk::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    // copied from https://github.com/ZhangJingrong/gpu_topK_benchmark/include/raft_topk.cuh
    // (updated)
    raft::matrix::detail::select::radix::select_k<
        float, std::int32_t, 11, 512, raft::matrix::detail::select::dense_layout<std::int32_t>>(
        *resources_, in_dist.data(), static_cast<std::int32_t*>(nullptr), in_dist.size(0),
        in_dist.size(1), k(), out_dist.data(), out_label.data(),
        true,                                 // select_min
        false,                                // fused_last_filter
        static_cast<std::int32_t*>(nullptr)); // not used in this case

    cuda_stream::make_default().sync();
}
