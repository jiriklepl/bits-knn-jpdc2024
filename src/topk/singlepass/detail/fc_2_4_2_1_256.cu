#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(2, 4, 2, 1, 256, 1, 256);
DECL_FC_KERNEL(2, 4, 2, 2, 128, 1, 256);
DECL_FC_KERNEL(2, 4, 2, 4, 64, 1, 256);