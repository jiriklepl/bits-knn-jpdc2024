#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(8, 16, 1, 1, 256, 4, 32);
DECL_FC_KERNEL(8, 16, 1, 2, 128, 4, 32);
DECL_FC_KERNEL(8, 16, 1, 4, 64, 4, 32);
