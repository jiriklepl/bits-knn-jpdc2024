#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(16, 4, 1, 1, 256, 1, 64);
DECL_FC_KERNEL(16, 4, 1, 2, 128, 1, 64);
DECL_FC_KERNEL(16, 4, 1, 4, 64, 1, 64);
