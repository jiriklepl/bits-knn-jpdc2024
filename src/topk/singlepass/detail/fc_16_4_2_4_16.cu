#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(16, 4, 2, 1, 256, 4, 16);
DECL_FC_KERNEL(16, 4, 2, 2, 128, 4, 16);
DECL_FC_KERNEL(16, 4, 2, 4, 64, 4, 16);
