#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(16, 8, 4, 1, 256, 1, 4);
DECL_FC_KERNEL(16, 8, 4, 2, 128, 1, 4);
DECL_FC_KERNEL(16, 8, 4, 4, 64, 1, 4);
