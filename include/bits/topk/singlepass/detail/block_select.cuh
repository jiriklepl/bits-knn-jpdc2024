#ifndef BITS_TOPK_SINGLEPASS_DETAIL_BLOCK_SELECT_CUH_
#define BITS_TOPK_SINGLEPASS_DETAIL_BLOCK_SELECT_CUH_

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <faiss/gpu/utils/BlockSelectKernel.cuh>

#include "bits/cuch.hpp"

#include "bits/topk/singlepass/block_select_runner.cuh"

template <int BLOCK_SIZE, int THREAD_QUEUE_SIZE, int K>
void block_select_runner::operator()()
{
    const auto block_count = dist_tensor.getSize(0);

    if (block_size != BLOCK_SIZE || thread_queue_size != THREAD_QUEUE_SIZE || k != K)
    {
        throw std::runtime_error("Invalid parameters for block_select");
    }

    faiss::gpu::blockSelect<float, std::int32_t, false, K, THREAD_QUEUE_SIZE, BLOCK_SIZE>
        <<<block_count, BLOCK_SIZE>>>(dist_tensor, out_dist_tensor, out_label_tensor,
                                      std::numeric_limits<float>::infinity(), -1, k);
    CUCH(cudaGetLastError());
}

#endif // BITS_TOPK_SINGLEPASS_DETAIL_BLOCK_SELECT_CUH_
