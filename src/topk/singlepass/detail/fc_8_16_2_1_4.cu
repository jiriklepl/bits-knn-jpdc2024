#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(8, 16, 2, 1, 256, 1, 4);
DECL_FC_KERNEL(8, 16, 2, 2, 128, 1, 4);
DECL_FC_KERNEL(8, 16, 2, 4, 64, 1, 4);
