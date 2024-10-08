#include "bits/topk/singlepass/block_select_runner.cuh"

template void block_select_runner::operator()<64, 2, 2048>();
template void block_select_runner::operator()<64, 3, 2048>();
template void block_select_runner::operator()<64, 4, 2048>();
template void block_select_runner::operator()<64, 5, 2048>();
template void block_select_runner::operator()<64, 6, 2048>();
template void block_select_runner::operator()<64, 7, 2048>();
template void block_select_runner::operator()<64, 8, 2048>();
template void block_select_runner::operator()<64, 9, 2048>();
template void block_select_runner::operator()<64, 10, 2048>();
