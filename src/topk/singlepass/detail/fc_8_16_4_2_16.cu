#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(8, 16, 4, 1, 256, 2, 16);
DECL_FC_KERNEL(8, 16, 4, 2, 128, 2, 16);
DECL_FC_KERNEL(8, 16, 4, 4, 64, 2, 16);
