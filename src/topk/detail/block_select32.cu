#include "bits/topk/singlepass/block_select_runner.cuh"

template void block_select_runner::operator()<128, 2, 32>();
template void block_select_runner::operator()<128, 3, 32>();
template void block_select_runner::operator()<128, 4, 32>();
template void block_select_runner::operator()<128, 5, 32>();
template void block_select_runner::operator()<128, 6, 32>();
template void block_select_runner::operator()<128, 7, 32>();
template void block_select_runner::operator()<128, 8, 32>();
template void block_select_runner::operator()<128, 9, 32>();
template void block_select_runner::operator()<128, 10, 32>();
