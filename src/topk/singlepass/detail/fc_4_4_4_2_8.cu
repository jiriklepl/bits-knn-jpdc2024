#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(4, 4, 4, 1, 256, 2, 8);
DECL_FC_KERNEL(4, 4, 4, 2, 128, 2, 8);
DECL_FC_KERNEL(4, 4, 4, 4, 64, 2, 8);
