#include "bits/topk/singlepass/detail/block_select.cuh"

template void block_select_runner::operator()<64, 2, 2048>();
template void block_select_runner::operator()<64, 3, 2048>();
template void block_select_runner::operator()<64, 4, 2048>();
template void block_select_runner::operator()<64, 5, 2048>();
template void block_select_runner::operator()<64, 6, 2048>();
template void block_select_runner::operator()<64, 7, 2048>();
template void block_select_runner::operator()<64, 8, 2048>();
template void block_select_runner::operator()<64, 9, 2048>();
template void block_select_runner::operator()<64, 10, 2048>();
template void block_select_runner::operator()<128, 2, 2048>();
template void block_select_runner::operator()<128, 3, 2048>();
template void block_select_runner::operator()<128, 4, 2048>();
template void block_select_runner::operator()<128, 5, 2048>();
template void block_select_runner::operator()<128, 6, 2048>();
template void block_select_runner::operator()<128, 7, 2048>();
template void block_select_runner::operator()<128, 8, 2048>();
template void block_select_runner::operator()<128, 9, 2048>();
template void block_select_runner::operator()<128, 10, 2048>();
