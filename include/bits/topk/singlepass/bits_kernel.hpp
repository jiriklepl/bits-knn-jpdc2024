#ifndef BITS_KERNEL_HPP
#define BITS_KERNEL_HPP

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/cuda_stream.hpp"

/** Bitonic select (bits) kernel (small k, multi-query -- one query per thread block)
 *
 * @tparam PREFETCH if true, the kernel will insert prefetch.global.L2 PTX instructions.
 * @tparam ADD_NORMS if true, the kernel will add @p norms to @p in_dist to finish distance
 * computation using cuBLAS.
 * @tparam BLOCK_SIZE number of threads in a thread block.
 * @tparam BATCH_SIZE number of reads per thread (we have to allocate an additional register for
 * each read).
 * @tparam K the number of values to find for each query.
 * @param[in] in_dist distance matrix.
 * @param[in] in_label label matrix (if it is nullptr, the kernel uses implicit indices as labels).
 * @param[out] out_dist top k distances for each query.
 * @param[out] out_label top k indices for each query.
 * @param[in] label_offsets this value will be multiplied by the block index and added to each label
 *                         (for the single-query adaptation of this kernel).
 * @param[in] norms computed norms of database vectors or nullptr if @p in_dist does not require
 *                  a postprocessing.
 */
template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE,
          std::size_t BATCH_SIZE, std::size_t K>
extern void run_bits_kernel(array_view<float, 2> in_dist, array_view<std::int32_t, 2> in_label,
                            array_view<float, 2> out_dist, array_view<std::int32_t, 2> out_label,
                            std::size_t k, const std::int32_t* label_offsets = nullptr,
                            const float* norms = nullptr,
                            cudaStream_t stream = cuda_stream::make_default().get());

#endif // BITS_KERNEL_HPP
