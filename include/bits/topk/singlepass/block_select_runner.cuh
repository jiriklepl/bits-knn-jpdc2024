#ifndef BLOCK_SELECT_RUNNER_CUH_
#define BLOCK_SELECT_RUNNER_CUH_

#include <cstdint>
#include <limits>

#include <faiss/gpu/utils/Tensor.cuh>

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
    void operator()();
};

#endif // BLOCK_SELECT_RUNNER_CUH_
