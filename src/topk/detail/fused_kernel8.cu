#include "bits/topk/singlepass/detail/fused_kernel.cuh"

template void fused_kernel_runner::operator()<8, 2, 4, 4>();
template void fused_kernel_runner::operator()<8, 4, 4, 4>();
template void fused_kernel_runner::operator()<8, 8, 4, 4>();

template void fused_kernel_runner::operator()<8, 2, 4, 8>();
template void fused_kernel_runner::operator()<8, 4, 4, 8>();
template void fused_kernel_runner::operator()<8, 8, 4, 8>();

template void fused_kernel_runner::operator()<8, 2, 4, 16>();
template void fused_kernel_runner::operator()<8, 4, 4, 16>();
template void fused_kernel_runner::operator()<8, 8, 4, 16>();
