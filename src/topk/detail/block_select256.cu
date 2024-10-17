#include "bits/topk/singlepass/detail/block_select.cuh"

template void block_select_runner::operator()<64, 2, 256>();
template void block_select_runner::operator()<64, 3, 256>();
template void block_select_runner::operator()<64, 4, 256>();
template void block_select_runner::operator()<64, 5, 256>();
template void block_select_runner::operator()<64, 6, 256>();
template void block_select_runner::operator()<64, 7, 256>();
template void block_select_runner::operator()<64, 8, 256>();
template void block_select_runner::operator()<64, 9, 256>();
template void block_select_runner::operator()<64, 10, 256>();
template void block_select_runner::operator()<128, 2, 256>();
template void block_select_runner::operator()<128, 3, 256>();
template void block_select_runner::operator()<128, 4, 256>();
template void block_select_runner::operator()<128, 5, 256>();
template void block_select_runner::operator()<128, 6, 256>();
template void block_select_runner::operator()<128, 7, 256>();
template void block_select_runner::operator()<128, 8, 256>();
template void block_select_runner::operator()<128, 9, 256>();
template void block_select_runner::operator()<128, 10, 256>();
