#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"
DECL_FC_KERNEL(4, 8, 2, 1, 256, 4, 64);
DECL_FC_KERNEL(4, 8, 2, 2, 128, 4, 64);
DECL_FC_KERNEL(4, 8, 2, 4, 64, 4, 64);
