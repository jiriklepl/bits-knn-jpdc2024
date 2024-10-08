#include "bits/topk/singlepass/fused_tc_kernel.cuh"

template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<2, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<4, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<8, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<16, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<32, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<64, 512>();
template void fused_tc_kernel_runner<fused_tc_kernel_bfloat16_policy>::operator()<128, 512>();
