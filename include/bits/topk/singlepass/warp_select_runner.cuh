#ifndef BITS_TOPK_SINGLEPASS_WARP_SELECT_RUNNER_CUH_
#define BITS_TOPK_SINGLEPASS_WARP_SELECT_RUNNER_CUH_

#include <cstdint>

#include <faiss/gpu/utils/Tensor.cuh>

struct warp_select_runner
{
    faiss::gpu::Tensor<float, 2, true> dist_tensor;
    faiss::gpu::Tensor<float, 2, true> out_dist_tensor;
    faiss::gpu::Tensor<std::int32_t, 2, true> out_label_tensor;
    int block_size;
    int thread_queue_size;
    int k;

    template <int BLOCK_SIZE, int THREAD_QUEUE_SIZE, int K>
    void operator()();
};

#endif // BITS_TOPK_SINGLEPASS_WARP_SELECT_RUNNER_CUH_
