#include "bits/topk/singlepass/detail/fused_tc_kernel.cuh"

template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<2, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<4, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<8, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<16, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<32, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<64, 256>();
template void fused_tc_kernel_runner<fused_tc_double_policy>::operator()<128, 256>();
