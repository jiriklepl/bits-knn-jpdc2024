#ifndef DETAIL_WARP_SELECT_CUH_
#define DETAIL_WARP_SELECT_CUH_

#include <cstdint>
#include <limits>

#include <faiss/gpu/utils/WarpSelectKernel.cuh>

#include "bits/topk/singlepass/warp_select_runner.cuh"

template <int BLOCK_SIZE, int THREAD_QUEUE_SIZE, int K>
void warp_select_runner::operator()()
{
    constexpr std::size_t WARP_SIZE = 32;
    constexpr std::size_t QUERIES_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    const auto block_count = (dist_tensor.getSize(0) + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;

    if (block_size != BLOCK_SIZE || thread_queue_size != THREAD_QUEUE_SIZE || k != K)
    {
        throw std::runtime_error("Invalid parameters for warp_select");
    }

    faiss::gpu::warpSelect<float, std::int32_t, false, K, THREAD_QUEUE_SIZE, BLOCK_SIZE>
        <<<block_count, BLOCK_SIZE>>>(dist_tensor, out_dist_tensor, out_label_tensor,
                                      std::numeric_limits<float>::infinity(), -1, k);
}

#endif // DETAIL_WARP_SELECT_CUH_