#ifndef BLOCK_SELECT_RUNNER_CUH_
#define BLOCK_SELECT_RUNNER_CUH_

#include <cstdint>
#include <limits>

#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>

/** The Block Select kernel from the FAISS library.
 */
struct block_select_runner
{
    /** Matrix with computed distances.
     */
    faiss::gpu::Tensor<float, 2, true> dist_tensor;

    /** Matrix with the output (top k distances for each query).
     */
    faiss::gpu::Tensor<float, 2, true> out_dist_tensor;

    /** Matrix with the output (top k labels for each query).
     */
    faiss::gpu::Tensor<std::int32_t, 2, true> out_label_tensor;

    /** Number of threads in each thread block.
     */
    int block_size;

    /** Size of each thread queue.
     */
    int thread_queue_size;

    /** Size of the output.
     */
    int k;

    template <int BLOCK_SIZE, int THREAD_QUEUE_SIZE, int K>
    void operator()()
    {
        const auto block_count = dist_tensor.getSize(0);

        if (block_size != BLOCK_SIZE || thread_queue_size != THREAD_QUEUE_SIZE || k != K)
        {
            throw std::runtime_error("Invalid parameters for block_select");
        }

        // limit the block size for the largest K (there is not enough shared memory)
        constexpr std::size_t ACTUAL_BLOCK_SIZE = K <= 1024 ? BLOCK_SIZE : 64;

        faiss::gpu::blockSelect<float, std::int32_t, false, K, THREAD_QUEUE_SIZE, ACTUAL_BLOCK_SIZE>
            <<<block_count, ACTUAL_BLOCK_SIZE>>>(dist_tensor, out_dist_tensor, out_label_tensor,
                                                 std::numeric_limits<float>::infinity(), -1, k);
    }
};

extern template void block_select_runner::operator()<128, 2, 32>();
extern template void block_select_runner::operator()<128, 3, 32>();
extern template void block_select_runner::operator()<128, 4, 32>();
extern template void block_select_runner::operator()<128, 5, 32>();
extern template void block_select_runner::operator()<128, 6, 32>();
extern template void block_select_runner::operator()<128, 7, 32>();
extern template void block_select_runner::operator()<128, 8, 32>();
extern template void block_select_runner::operator()<128, 9, 32>();
extern template void block_select_runner::operator()<128, 10, 32>();

extern template void block_select_runner::operator()<64, 2, 2048>();
extern template void block_select_runner::operator()<64, 3, 2048>();
extern template void block_select_runner::operator()<64, 4, 2048>();
extern template void block_select_runner::operator()<64, 5, 2048>();
extern template void block_select_runner::operator()<64, 6, 2048>();
extern template void block_select_runner::operator()<64, 7, 2048>();
extern template void block_select_runner::operator()<64, 8, 2048>();
extern template void block_select_runner::operator()<64, 9, 2048>();
extern template void block_select_runner::operator()<64, 10, 2048>();

extern template void block_select_runner::operator()<128, 2, 128>();
extern template void block_select_runner::operator()<128, 3, 128>();
extern template void block_select_runner::operator()<128, 4, 128>();
extern template void block_select_runner::operator()<128, 5, 128>();
extern template void block_select_runner::operator()<128, 6, 128>();
extern template void block_select_runner::operator()<128, 7, 128>();
extern template void block_select_runner::operator()<128, 8, 128>();
extern template void block_select_runner::operator()<128, 9, 128>();
extern template void block_select_runner::operator()<128, 10, 128>();

extern template void block_select_runner::operator()<128, 2, 256>();
extern template void block_select_runner::operator()<128, 3, 256>();
extern template void block_select_runner::operator()<128, 4, 256>();
extern template void block_select_runner::operator()<128, 5, 256>();
extern template void block_select_runner::operator()<128, 6, 256>();
extern template void block_select_runner::operator()<128, 7, 256>();
extern template void block_select_runner::operator()<128, 8, 256>();
extern template void block_select_runner::operator()<128, 9, 256>();
extern template void block_select_runner::operator()<128, 10, 256>();

extern template void block_select_runner::operator()<128, 2, 512>();
extern template void block_select_runner::operator()<128, 3, 512>();
extern template void block_select_runner::operator()<128, 4, 512>();
extern template void block_select_runner::operator()<128, 5, 512>();
extern template void block_select_runner::operator()<128, 6, 512>();
extern template void block_select_runner::operator()<128, 7, 512>();
extern template void block_select_runner::operator()<128, 8, 512>();
extern template void block_select_runner::operator()<128, 9, 512>();
extern template void block_select_runner::operator()<128, 10, 512>();

extern template void block_select_runner::operator()<128, 2, 1024>();
extern template void block_select_runner::operator()<128, 3, 1024>();
extern template void block_select_runner::operator()<128, 4, 1024>();
extern template void block_select_runner::operator()<128, 5, 1024>();
extern template void block_select_runner::operator()<128, 6, 1024>();
extern template void block_select_runner::operator()<128, 7, 1024>();
extern template void block_select_runner::operator()<128, 8, 1024>();
extern template void block_select_runner::operator()<128, 9, 1024>();
extern template void block_select_runner::operator()<128, 10, 1024>();

extern template void block_select_runner::operator()<64, 2, 2048>();
extern template void block_select_runner::operator()<64, 3, 2048>();
extern template void block_select_runner::operator()<64, 4, 2048>();
extern template void block_select_runner::operator()<64, 5, 2048>();
extern template void block_select_runner::operator()<64, 6, 2048>();
extern template void block_select_runner::operator()<64, 7, 2048>();
extern template void block_select_runner::operator()<64, 8, 2048>();
extern template void block_select_runner::operator()<64, 9, 2048>();
extern template void block_select_runner::operator()<64, 10, 2048>();

#endif // BLOCK_SELECT_RUNNER_CUH_
