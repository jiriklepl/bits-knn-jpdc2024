#include "bits/topk/singlepass/warp_select_runner.cuh"

template void warp_select_runner::operator()<128, 2, 256>();
template void warp_select_runner::operator()<128, 3, 256>();
template void warp_select_runner::operator()<128, 4, 256>();
template void warp_select_runner::operator()<128, 5, 256>();
template void warp_select_runner::operator()<128, 6, 256>();
template void warp_select_runner::operator()<128, 7, 256>();
template void warp_select_runner::operator()<128, 8, 256>();
template void warp_select_runner::operator()<128, 9, 256>();
template void warp_select_runner::operator()<128, 10, 256>();
