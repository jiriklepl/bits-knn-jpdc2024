#include "bits/topk/singlepass/detail/warp_select.cuh"

template void warp_select_runner::operator()<64, 2, 512>();
template void warp_select_runner::operator()<64, 3, 512>();
template void warp_select_runner::operator()<64, 4, 512>();
template void warp_select_runner::operator()<64, 5, 512>();
template void warp_select_runner::operator()<64, 6, 512>();
template void warp_select_runner::operator()<64, 7, 512>();
template void warp_select_runner::operator()<64, 8, 512>();
template void warp_select_runner::operator()<64, 9, 512>();
template void warp_select_runner::operator()<64, 10, 512>();
template void warp_select_runner::operator()<128, 2, 512>();
template void warp_select_runner::operator()<128, 3, 512>();
template void warp_select_runner::operator()<128, 4, 512>();
template void warp_select_runner::operator()<128, 5, 512>();
template void warp_select_runner::operator()<128, 6, 512>();
template void warp_select_runner::operator()<128, 7, 512>();
template void warp_select_runner::operator()<128, 8, 512>();
template void warp_select_runner::operator()<128, 9, 512>();
template void warp_select_runner::operator()<128, 10, 512>();
