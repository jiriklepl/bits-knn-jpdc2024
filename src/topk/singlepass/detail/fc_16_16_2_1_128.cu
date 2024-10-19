#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(16, 16, 2, 1, 256, 1, 128);
DECL_FC_KERNEL(16, 16, 2, 2, 128, 1, 128);
DECL_FC_KERNEL(16, 16, 2, 4, 64, 1, 128);
