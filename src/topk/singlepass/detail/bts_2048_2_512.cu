#include "bits/topk/singlepass/detail/bits_kernel.cuh"

DECL_BITS_KERNEL(float, std::int32_t, false, 512, 2, 2048);
DECL_BITS_KERNEL(float, std::int32_t, true, 512, 2, 2048);
