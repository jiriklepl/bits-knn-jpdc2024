#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(2, 8, 2, 1, 256, 4, 4);
DECL_FC_KERNEL(2, 8, 2, 2, 128, 4, 4);
DECL_FC_KERNEL(2, 8, 2, 4, 64, 4, 4);
