#include "bits/topk/singlepass/detail/block_select.cuh"

template void block_select_runner::operator()<64, 2, 1024>();
template void block_select_runner::operator()<64, 3, 1024>();
template void block_select_runner::operator()<64, 4, 1024>();
template void block_select_runner::operator()<64, 5, 1024>();
template void block_select_runner::operator()<64, 6, 1024>();
template void block_select_runner::operator()<64, 7, 1024>();
template void block_select_runner::operator()<64, 8, 1024>();
template void block_select_runner::operator()<64, 9, 1024>();
template void block_select_runner::operator()<64, 10, 1024>();
template void block_select_runner::operator()<128, 2, 1024>();
template void block_select_runner::operator()<128, 3, 1024>();
template void block_select_runner::operator()<128, 4, 1024>();
template void block_select_runner::operator()<128, 5, 1024>();
template void block_select_runner::operator()<128, 6, 1024>();
template void block_select_runner::operator()<128, 7, 1024>();
template void block_select_runner::operator()<128, 8, 1024>();
template void block_select_runner::operator()<128, 9, 1024>();
template void block_select_runner::operator()<128, 10, 1024>();
