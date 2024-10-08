#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(2, 4, 1, 1, 256, 4, 32);
DECL_FC_KERNEL(2, 4, 1, 2, 128, 4, 32);
DECL_FC_KERNEL(2, 4, 1, 4, 64, 4, 32);
