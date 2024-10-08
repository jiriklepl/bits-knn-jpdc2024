#include "bits/topk/singlepass/fused_tc_kernel.cuh"

template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<2, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<4, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<8, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<16, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<32, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<64, 128>();
template void fused_tc_kernel_runner<fused_tc_kernel_half_policy>::operator()<128, 128>();
