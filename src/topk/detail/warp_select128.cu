#include "bits/topk/singlepass/detail/warp_select.cuh"

template void warp_select_runner::operator()<64, 2, 128>();
template void warp_select_runner::operator()<64, 3, 128>();
template void warp_select_runner::operator()<64, 4, 128>();
template void warp_select_runner::operator()<64, 5, 128>();
template void warp_select_runner::operator()<64, 6, 128>();
template void warp_select_runner::operator()<64, 7, 128>();
template void warp_select_runner::operator()<64, 8, 128>();
template void warp_select_runner::operator()<64, 9, 128>();
template void warp_select_runner::operator()<64, 10, 128>();
template void warp_select_runner::operator()<128, 2, 128>();
template void warp_select_runner::operator()<128, 3, 128>();
template void warp_select_runner::operator()<128, 4, 128>();
template void warp_select_runner::operator()<128, 5, 128>();
template void warp_select_runner::operator()<128, 6, 128>();
template void warp_select_runner::operator()<128, 7, 128>();
template void warp_select_runner::operator()<128, 8, 128>();
template void warp_select_runner::operator()<128, 9, 128>();
template void warp_select_runner::operator()<128, 10, 128>();
