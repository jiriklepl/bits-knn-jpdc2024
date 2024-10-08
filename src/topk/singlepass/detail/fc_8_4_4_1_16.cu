#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(8, 4, 4, 1, 256, 1, 16);
DECL_FC_KERNEL(8, 4, 4, 2, 128, 1, 16);
DECL_FC_KERNEL(8, 4, 4, 4, 64, 1, 16);
