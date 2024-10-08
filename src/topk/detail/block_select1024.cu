#include "bits/topk/singlepass/block_select_runner.cuh"

template void block_select_runner::operator()<128, 2, 1024>();
template void block_select_runner::operator()<128, 3, 1024>();
template void block_select_runner::operator()<128, 4, 1024>();
template void block_select_runner::operator()<128, 5, 1024>();
template void block_select_runner::operator()<128, 6, 1024>();
template void block_select_runner::operator()<128, 7, 1024>();
template void block_select_runner::operator()<128, 8, 1024>();
template void block_select_runner::operator()<128, 9, 1024>();
template void block_select_runner::operator()<128, 10, 1024>();
