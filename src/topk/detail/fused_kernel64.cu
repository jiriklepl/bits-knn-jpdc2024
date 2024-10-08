#include "bits/topk/singlepass/fused_kernel.cuh"

template void fused_kernel_runner::operator()<64, 2, 4, 4>();
template void fused_kernel_runner::operator()<64, 4, 4, 4>();
template void fused_kernel_runner::operator()<64, 8, 4, 4>();

template void fused_kernel_runner::operator()<64, 2, 4, 8>();
template void fused_kernel_runner::operator()<64, 4, 4, 8>();
template void fused_kernel_runner::operator()<64, 8, 4, 8>();

template void fused_kernel_runner::operator()<64, 2, 4, 16>();
template void fused_kernel_runner::operator()<64, 4, 4, 16>();
template void fused_kernel_runner::operator()<64, 8, 4, 16>();
