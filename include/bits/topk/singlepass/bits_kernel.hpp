#ifndef BITS_TOPK_SINGLEPASS_BITS_KERNEL_HPP
#define BITS_TOPK_SINGLEPASS_BITS_KERNEL_HPP

#include <cstddef>

#include "bits/array_view.hpp"
#include "bits/cuda_stream.hpp"

/** Bitonic select (bits) kernel (small k, multi-query -- one query per thread block)
 *
 * @tparam PREFETCH if true, the kernel will insert prefetch.global.L2 PTX instructions.
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam BATCH_SIZE number of elements to load for each thread in a single iteration.
 * @tparam K the number of values to find for each query.
 * @param[in] in_dist distance matrix.
 * @param[in] in_label label matrix (if it is nullptr, the kernel uses implicit indices as labels).
 * @param[out] out_dist top k distances for each query.
 * @param[out] out_label top k indices for each query.
 * @param[in] label_offsets offsets to add to the labels (useful for single-query problems). nullptr
 * if not needed.
 */
template <class Value, class Idx, bool PREFETCH, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE,
          std::size_t K>
extern void run_bits_kernel(array_view<Value, 2> in_dist, array_view<Idx, 2> in_label,
                            array_view<Value, 2> out_dist, array_view<Idx, 2> out_label,
                            std::size_t k, const Idx* label_offsets = nullptr,
                            cudaStream_t stream = cuda_stream::make_default().get());

#endif // BITS_TOPK_SINGLEPASS_BITS_KERNEL_HPP
