#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(8, 8, 2, 1, 256, 2, 256);
DECL_FC_KERNEL(8, 8, 2, 2, 128, 2, 256);
DECL_FC_KERNEL(8, 8, 2, 4, 64, 2, 256);