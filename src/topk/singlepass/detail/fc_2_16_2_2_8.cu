#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(2, 16, 2, 1, 256, 2, 8);
DECL_FC_KERNEL(2, 16, 2, 2, 128, 2, 8);
DECL_FC_KERNEL(2, 16, 2, 4, 64, 2, 8);
