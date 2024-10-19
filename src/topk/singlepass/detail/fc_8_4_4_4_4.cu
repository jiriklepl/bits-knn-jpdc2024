#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(8, 4, 4, 1, 256, 4, 4);
DECL_FC_KERNEL(8, 4, 4, 2, 128, 4, 4);
DECL_FC_KERNEL(8, 4, 4, 4, 64, 4, 4);
