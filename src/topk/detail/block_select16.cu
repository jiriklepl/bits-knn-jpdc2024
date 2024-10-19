#include "bits/topk/singlepass/detail/block_select.cuh"

template void block_select_runner::operator()<64, 2, 16>();
template void block_select_runner::operator()<64, 3, 16>();
template void block_select_runner::operator()<64, 4, 16>();
template void block_select_runner::operator()<64, 5, 16>();
template void block_select_runner::operator()<64, 6, 16>();
template void block_select_runner::operator()<64, 7, 16>();
template void block_select_runner::operator()<64, 8, 16>();
template void block_select_runner::operator()<64, 9, 16>();
template void block_select_runner::operator()<64, 10, 16>();
template void block_select_runner::operator()<128, 2, 16>();
template void block_select_runner::operator()<128, 3, 16>();
template void block_select_runner::operator()<128, 4, 16>();
template void block_select_runner::operator()<128, 5, 16>();
template void block_select_runner::operator()<128, 6, 16>();
template void block_select_runner::operator()<128, 7, 16>();
template void block_select_runner::operator()<128, 8, 16>();
template void block_select_runner::operator()<128, 9, 16>();
template void block_select_runner::operator()<128, 10, 16>();
