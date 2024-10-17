#include "bits/topk/singlepass/detail/block_select.cuh"

template void block_select_runner::operator()<64, 2, 64>();
template void block_select_runner::operator()<64, 3, 64>();
template void block_select_runner::operator()<64, 4, 64>();
template void block_select_runner::operator()<64, 5, 64>();
template void block_select_runner::operator()<64, 6, 64>();
template void block_select_runner::operator()<64, 7, 64>();
template void block_select_runner::operator()<64, 8, 64>();
template void block_select_runner::operator()<64, 9, 64>();
template void block_select_runner::operator()<64, 10, 64>();
template void block_select_runner::operator()<128, 2, 64>();
template void block_select_runner::operator()<128, 3, 64>();
template void block_select_runner::operator()<128, 4, 64>();
template void block_select_runner::operator()<128, 5, 64>();
template void block_select_runner::operator()<128, 6, 64>();
template void block_select_runner::operator()<128, 7, 64>();
template void block_select_runner::operator()<128, 8, 64>();
template void block_select_runner::operator()<128, 9, 64>();
template void block_select_runner::operator()<128, 10, 64>();
