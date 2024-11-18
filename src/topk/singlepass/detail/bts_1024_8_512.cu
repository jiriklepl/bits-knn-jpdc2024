#include "bits/topk/singlepass/detail/bits_kernel.cuh"

DECL_BITS_KERNEL(float, std::int32_t, false, 512, 8, 1024);
DECL_BITS_KERNEL(float, std::int32_t, true, 512, 8, 1024);
