#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(16, 4, 1, 1, 256, 4, 128);
DECL_FC_KERNEL(16, 4, 1, 2, 128, 4, 128);
DECL_FC_KERNEL(16, 4, 1, 4, 64, 4, 128);
