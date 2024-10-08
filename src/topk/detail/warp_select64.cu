#include "bits/topk/singlepass/warp_select_runner.cuh"

template void warp_select_runner::operator()<128, 2, 64>();
template void warp_select_runner::operator()<128, 3, 64>();
template void warp_select_runner::operator()<128, 4, 64>();
template void warp_select_runner::operator()<128, 5, 64>();
template void warp_select_runner::operator()<128, 6, 64>();
template void warp_select_runner::operator()<128, 7, 64>();
template void warp_select_runner::operator()<128, 8, 64>();
template void warp_select_runner::operator()<128, 9, 64>();
template void warp_select_runner::operator()<128, 10, 64>();
