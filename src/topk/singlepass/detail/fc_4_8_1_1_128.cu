#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"

DECL_FC_KERNEL(4, 8, 1, 1, 256, 1, 128);
DECL_FC_KERNEL(4, 8, 1, 2, 128, 1, 128);
DECL_FC_KERNEL(4, 8, 1, 4, 64, 1, 128);
