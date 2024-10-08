#include <cstddef>
#include <cstdint>

#include <raft/core/device_resources.hpp>
#include <raft/matrix/detail/select_radix.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/multipass/air_topk.hpp"

namespace
{

// copied from external/gpu_topK_benchmark/include/raft_topk.cuh
// https://github.com/rapidsai/raft/blob/branch-22.06/cpp/bench/common/benchmark.hpp
struct using_pool_memory_res
{
private:
    [[maybe_unused]] rmm::mr::device_memory_resource* orig_res_;
    [[maybe_unused]] rmm::mr::cuda_memory_resource cuda_res_;
    [[maybe_unused]] rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_res_;

public:
    using_pool_memory_res(size_t initial_size, size_t max_size)
        : orig_res_(rmm::mr::get_current_device_resource()),
          pool_res_(&cuda_res_, initial_size, max_size)
    {
        rmm::mr::set_current_device_resource(&pool_res_);
    }

    using_pool_memory_res()
        : using_pool_memory_res(size_t(1) << size_t(30), size_t(16) << size_t(30))
    {
    }

    ~using_pool_memory_res() { rmm::mr::set_current_device_resource(orig_res_); }

    using_pool_memory_res(const using_pool_memory_res&) = delete;
    using_pool_memory_res& operator=(const using_pool_memory_res&) = delete;
    using_pool_memory_res(using_pool_memory_res&&) = delete;
    using_pool_memory_res& operator=(using_pool_memory_res&&) = delete;
};

[[maybe_unused]] const using_pool_memory_res rmm_res{};

} // namespace

void air_topk::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    raft::device_resources resources{};

    // copied from external/gpu_topK_benchmark/include/raft_topk.cuh (updated)
    raft::matrix::detail::select::radix::select_k<float, std::int32_t, 11, 512>(
        resources, in_dist.data(), static_cast<std::int32_t*>(nullptr), in_dist.size(0),
        in_dist.size(1), k(), out_dist.data(), out_label.data(),
        true,                                 // select_min
        false,                                // fused_last_filter
        static_cast<std::int32_t*>(nullptr)); // not used in this case

    cuda_stream::make_default().sync();
}
