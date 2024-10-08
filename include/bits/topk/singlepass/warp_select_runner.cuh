#ifndef WARP_SELECT_RUNNER_CUH_
#define WARP_SELECT_RUNNER_CUH_

#include <cstdint>
#include <limits>

#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>

struct warp_select_runner
{
    faiss::gpu::Tensor<float, 2, true> dist_tensor;
    faiss::gpu::Tensor<float, 2, true> out_dist_tensor;
    faiss::gpu::Tensor<std::int32_t, 2, true> out_label_tensor;
    int block_size;
    int thread_queue_size;
    int k;

    template <int BLOCK_SIZE, int THREAD_QUEUE_SIZE, int K>
    void operator()()
    {
        constexpr std::size_t WARP_SIZE = 32;
        constexpr std::size_t QUERIES_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
        const auto block_count =
            (dist_tensor.getSize(0) + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;

        if (block_size != BLOCK_SIZE || thread_queue_size != THREAD_QUEUE_SIZE || k != K)
        {
            throw std::runtime_error("Invalid parameters for warp_select");
        }

        faiss::gpu::warpSelect<float, std::int32_t, false, K, THREAD_QUEUE_SIZE, BLOCK_SIZE>
            <<<block_count, BLOCK_SIZE>>>(dist_tensor, out_dist_tensor, out_label_tensor,
                                          std::numeric_limits<float>::infinity(), -1, k);
    }
};

extern template void warp_select_runner::operator()<128, 2, 32>();
extern template void warp_select_runner::operator()<128, 3, 32>();
extern template void warp_select_runner::operator()<128, 4, 32>();
extern template void warp_select_runner::operator()<128, 5, 32>();
extern template void warp_select_runner::operator()<128, 6, 32>();
extern template void warp_select_runner::operator()<128, 7, 32>();
extern template void warp_select_runner::operator()<128, 8, 32>();
extern template void warp_select_runner::operator()<128, 9, 32>();
extern template void warp_select_runner::operator()<128, 10, 32>();

extern template void warp_select_runner::operator()<128, 2, 64>();
extern template void warp_select_runner::operator()<128, 3, 64>();
extern template void warp_select_runner::operator()<128, 4, 64>();
extern template void warp_select_runner::operator()<128, 5, 64>();
extern template void warp_select_runner::operator()<128, 6, 64>();
extern template void warp_select_runner::operator()<128, 7, 64>();
extern template void warp_select_runner::operator()<128, 8, 64>();
extern template void warp_select_runner::operator()<128, 9, 64>();
extern template void warp_select_runner::operator()<128, 10, 64>();

extern template void warp_select_runner::operator()<128, 2, 128>();
extern template void warp_select_runner::operator()<128, 3, 128>();
extern template void warp_select_runner::operator()<128, 4, 128>();
extern template void warp_select_runner::operator()<128, 5, 128>();
extern template void warp_select_runner::operator()<128, 6, 128>();
extern template void warp_select_runner::operator()<128, 7, 128>();
extern template void warp_select_runner::operator()<128, 8, 128>();
extern template void warp_select_runner::operator()<128, 9, 128>();
extern template void warp_select_runner::operator()<128, 10, 128>();

extern template void warp_select_runner::operator()<128, 2, 256>();
extern template void warp_select_runner::operator()<128, 3, 256>();
extern template void warp_select_runner::operator()<128, 4, 256>();
extern template void warp_select_runner::operator()<128, 5, 256>();
extern template void warp_select_runner::operator()<128, 6, 256>();
extern template void warp_select_runner::operator()<128, 7, 256>();
extern template void warp_select_runner::operator()<128, 8, 256>();
extern template void warp_select_runner::operator()<128, 9, 256>();
extern template void warp_select_runner::operator()<128, 10, 256>();

extern template void warp_select_runner::operator()<128, 2, 512>();
extern template void warp_select_runner::operator()<128, 3, 512>();
extern template void warp_select_runner::operator()<128, 4, 512>();
extern template void warp_select_runner::operator()<128, 5, 512>();
extern template void warp_select_runner::operator()<128, 6, 512>();
extern template void warp_select_runner::operator()<128, 7, 512>();
extern template void warp_select_runner::operator()<128, 8, 512>();
extern template void warp_select_runner::operator()<128, 9, 512>();
extern template void warp_select_runner::operator()<128, 10, 512>();

extern template void warp_select_runner::operator()<128, 2, 1024>();
extern template void warp_select_runner::operator()<128, 3, 1024>();
extern template void warp_select_runner::operator()<128, 4, 1024>();
extern template void warp_select_runner::operator()<128, 5, 1024>();
extern template void warp_select_runner::operator()<128, 6, 1024>();
extern template void warp_select_runner::operator()<128, 7, 1024>();
extern template void warp_select_runner::operator()<128, 8, 1024>();
extern template void warp_select_runner::operator()<128, 9, 1024>();
extern template void warp_select_runner::operator()<128, 10, 1024>();

extern template void warp_select_runner::operator()<128, 2, 2048>();
extern template void warp_select_runner::operator()<128, 3, 2048>();
extern template void warp_select_runner::operator()<128, 4, 2048>();
extern template void warp_select_runner::operator()<128, 5, 2048>();
extern template void warp_select_runner::operator()<128, 6, 2048>();
extern template void warp_select_runner::operator()<128, 7, 2048>();
extern template void warp_select_runner::operator()<128, 8, 2048>();
extern template void warp_select_runner::operator()<128, 9, 2048>();
extern template void warp_select_runner::operator()<128, 10, 2048>();

#endif // WARP_SELECT_RUNNER_CUH_
