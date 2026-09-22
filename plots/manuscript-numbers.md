Ratios use mean post-warmup times; summaries weight workloads equally.
Bandwidth follows scripts/utils.py. Parameter values are the recorded CSV fields.

| Metric | V100 (volta05) | L40 (ampere01) | A100 (ampere02) | H100 (hopper01) | RTX PRO 6000 Blackwell (bw01) |
|---|---|---|---|---|---|
| BITS speedup vs best competitor: mean / min / max | 2.13× / 1.55× / 2.65× | 1.29× / 0.98× / 2.24× | 1.77× / 0.96× / 2.23× | 1.85× / 1.12× / 2.38× | 1.41× / 0.99× / 2.05× |
| BITS peak-speedup workload | n=1048576, q=1024, d=1, k=512 | n=4194304, q=256, d=1, k=2048 | n=262144, q=4096, d=1, k=256 | n=4194304, q=256, d=1, k=256 | n=4194304, q=256, d=1, k=1024 |
| BITS minimum-speedup workload | n=262144, q=4096, d=1, k=32 | n=262144, q=4096, d=1, k=32 | n=262144, q=4096, d=1, k=2048 | n=262144, q=4096, d=1, k=2048 | n=16777216, q=64, d=1, k=64 |
| BITS speedup at q=64: mean | 1.97× | 1.29× | 1.42× | 1.58× | 1.38× |
| BITS wins vs best competitor | 28/28 | 23/28 | 27/28 | 28/28 | 27/28 |
| BITS throughput: peak (distances/s) | 2.1418e+11 | 2.0743e+11 | 4.1097e+11 | 4.7894e+11 | 3.7381e+11 |
| BITS bandwidth utilization: mean / peak | 82.08% / 88.65% | 88.61% / 89.44% | 64.04% / 79.12% | 74.19% / 87.50% | 84.79% / 87.20% |
| Prefetch speedup gain: mean / min / max | 1.75% / -7.39% / 11.64% | -0.02% / -0.44% / 1.14% | 5.89% / -2.30% / 23.90% | 4.73% / -3.49% / 21.15% | 6.28% / -0.42% / 23.97% |
| Timing variation (std/mean): maximum | 8.93% | 11.32% | 3.61% | 26.27% | 2.79% |
| Sort-in-registers speedup: mean / min / max | 2.86× / 1.86× / 10.28× | 2.43× / 1.41× / 8.89× | 2.72× / 1.54× / 10.23× | 2.97× / 1.65× / 11.88× | 3.08× / 1.67× / 10.86× |
| Sort-in-registers speedup: median | 2.18× | 1.81× | 2.11× | 2.08× | 2.32× |
| Sort-in-registers fastest | 28/28 | 26/28 | 28/28 | 25/28 | 25/28 |
| Fixed partial-bitonic: block; worst slowdown | 128; 45.52% | 128; 29.21% | 128; 33.80% | 128; 31.28% | 128; 41.21% |
| Fixed partial-bitonic-warp: block; worst slowdown | 128; 96.21% | 128; 44.16% | 128; 53.92% | 128; 29.25% | 128; 81.99% |
| Fixed partial-bitonic-warp-static: block; worst slowdown | 128; 83.58% | 128; 42.44% | 128; 57.99% | 128; 38.27% | 128; 74.19% |
| Fixed partial-bitonic-regs: block; worst slowdown | 128; 49.45% | 128; 30.37% | 128; 17.60% | 128; 21.49% | 128; 42.07% |
| Buffer speedup (ascending): mean / min / max | 5.48× / 2.61× / 7.00× | 5.21× / 3.50× / 7.52× | 6.48× / 5.02× / 9.52× | 5.86× / 4.23× / 8.13× | 3.83× / 2.00× / 6.73× |
| Buffer speedup (identity): mean / min / max | 5.24× / 2.52× / 6.77× | 5.30× / 3.63× / 7.55× | 6.12× / 4.86× / 9.03× | 5.56× / 4.10× / 7.77× | 3.77× / 1.83× / 6.81× |
| Buffer speedup (descending): mean / min / max | 0.77× / 0.57× / 0.88× | 0.72× / 0.48× / 0.88× | 0.82× / 0.65× / 0.89× | 0.80× / 0.67× / 0.92× | 0.69× / 0.39× / 0.86× |
| Fixed BITS parameters: block; items; degree | 512; 5,1,1; 1 | 512; 5,1,1; 1 | 512; 7,1,1; 1 | 256; 13,1,1; 1 | 512; 4,1,1; 1 |
| Fixed BITS worst slowdown | 16.43% | 1.72% | 27.11% | 31.60% | 6.66% |
| MAGMA-distance wins vs other plotted kernels | 25/28 | 24/28 | 24/28 | 27/28 | 22/28 |
| Distance parameters fixed per d: worst slowdown | 1.94% | 5.29% | 3.53% | 1.29% | 0.50% |
| Fused vs RAFT, d<=8: mean / min / max | 3.45× / 2.22× / 6.45× | 3.08× / 1.95× / 5.29× | 3.00× / 1.69× / 5.90× | 2.80× / 1.55× / 5.48× | 2.35× / 1.58× / 3.24× |
| Fused vs RAFT, d<=8: speedup >2× | 32/32 | 30/32 | 29/32 | 26/32 | 24/32 |
| Fused wins vs RAFT | 48/48 | 48/48 | 48/48 | 48/48 | 48/48 |
| Fused vs two-phase: mean / min / max | 2.73× / 1.47× / 4.06× | 5.45× / 2.88× / 8.62× | 1.91× / 0.84× / 3.20× | 1.78× / 0.91× / 2.85× | 3.04× / 1.48× / 5.01× |
| Fused wins vs two-phase | 72/72 | 72/72 | 65/72 | 65/72 | 72/72 |
| Fused wins vs two-phase, d=4 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| Fused wins vs two-phase, d=8 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| Fused wins vs two-phase, d=16 | 24/24 | 24/24 | 17/24 | 17/24 | 24/24 |
| RAFT throughput loss at k=64 vs k=8: mean / min / max | 48.36% / 40.45% / 58.14% | 39.17% / 37.86% / 40.59% | 46.47% / 42.31% / 54.53% | 46.83% / 42.48% / 54.33% | 13.97% / 10.94% / 16.66% |
| Fused throughput loss at k=256 vs k=8: mean / min / max | 24.92% / 14.41% / 31.43% | 13.03% / -7.28% / 25.42% | 11.78% / -1.28% / 27.63% | 8.18% / -1.36% / 19.86% | 6.21% / -7.91% / 24.30% |
| Fused wins vs BITS + zero computation, d=4 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| Fused wins vs BITS + zero computation, d=8 | 24/24 | 24/24 | 18/24 | 24/24 | 24/24 |
| Fused wins vs BITS + zero computation, d=16 | 24/24 | 24/24 | 5/24 | 12/24 | 24/24 |
| Two-phase distance matrix (GiB) | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| Fixed fused parameters: query tile; items tuple; degree | 1; 8,4,1; 2 | 2; 4,4,4; 1 | 2; 4,4,1; 4 | 1; 8,4,2; 1 | 2; 4,8,1; 4 |
| Fixed fused worst slowdown | 34.34% | 45.62% | 41.57% | 27.10% | 33.08% |
| Fixed fused vs RAFT: worst overhead (tuning/eval) | -14.30% | -6.43% | 18.47% | 1.67% | -4.40% |
| Fixed fused vs RAFT: worst workload | n=131072, q=8192, d=16, k=8 | n=131072, q=8192, d=16, k=8 | n=1048576, q=1024, d=16, k=8 | n=1048576, q=1024, d=16, k=8 | n=131072, q=8192, d=16, k=8 |
| Fixed fused wins vs two-phase (tuning/eval) | 72/72 | 72/72 | 60/72 | 60/72 | 72/72 |

| Run family | volta05 | ampere01 | ampere02 | hopper01 | bw01 |
|---|---|---|---|---|---|
| bitonic-sort | 108428 | 108429 | 35757456 | 35757470 | 35757463 |
| buffer | 108406, 108410, 108414 | 108404, 108409, 108415 | 35757457, 35757458, 35757459 | 35757471, 35757472, 35757473 | 35757464, 35757465, 35757466 |
| kselection | 108425 | 108426 | 35757460 | 35757474 | 35757467 |
| distances | 108421 | 108422 | 35757461 | 35757475 | 35757468 |
| fused | 108418 | 108419 | 35757462 | 35757476 | 35757469 |
| opt-bitonic-sort | 108290 | 108293 | 108291 | 101133 | 35746911 |
| opt-ipt | 108190, 108202, 108205 | 108187, 108199, 108203 | 108188, 108200, 108206 | 108189, 108201, 108204 | 35746903, 35746904, 35746905 |
| opt-distances | 108258 | 108256 | 108255 | 101154 | 35746923 |
| fused-cache-params | 108238, 108242, 108246 | 108241, 108245, 108248 | 108240, 108243, 108247 | 108239, 108244, 108249 | 35746908, 35746909, 35746910 |
