#include "bits/topk/singlepass/warp_select_runner.cuh"

template void warp_select_runner::operator()<128, 2, 2048>();
template void warp_select_runner::operator()<128, 3, 2048>();
template void warp_select_runner::operator()<128, 4, 2048>();
template void warp_select_runner::operator()<128, 5, 2048>();
template void warp_select_runner::operator()<128, 6, 2048>();
template void warp_select_runner::operator()<128, 7, 2048>();
template void warp_select_runner::operator()<128, 8, 2048>();
template void warp_select_runner::operator()<128, 9, 2048>();
template void warp_select_runner::operator()<128, 10, 2048>();
