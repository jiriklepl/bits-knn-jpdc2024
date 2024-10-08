#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(4, 16, 1, 1, 256, 1, 8);
DECL_FC_KERNEL(4, 16, 1, 2, 128, 1, 8);
DECL_FC_KERNEL(4, 16, 1, 4, 64, 1, 8);
