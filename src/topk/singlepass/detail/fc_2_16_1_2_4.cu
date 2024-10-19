#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(2, 16, 1, 1, 256, 2, 4);
DECL_FC_KERNEL(2, 16, 1, 2, 128, 2, 4);
DECL_FC_KERNEL(2, 16, 1, 4, 64, 2, 4);
