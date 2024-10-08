#ifndef BITS_GLOBAL_KERNEL_HPP_
#define BITS_GLOBAL_KERNEL_HPP_

#include <cstddef>
#include <cstdint>

#include "bits/array_view.hpp"

template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
extern void run_bits_global_kernel(array_view<float, 2> dist, array_view<float, 2> out_dist,
                                   array_view<std::int32_t, 2> out_label, std::size_t k);

#endif // BITS_GLOBAL_KERNEL_HPP_
