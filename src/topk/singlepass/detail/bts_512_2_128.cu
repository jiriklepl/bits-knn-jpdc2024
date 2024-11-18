#include "bits/topk/singlepass/detail/bits_kernel.cuh"

DECL_BITS_KERNEL(float, std::int32_t, false, 128, 2, 512);
DECL_BITS_KERNEL(float, std::int32_t, true, 128, 2, 512);
