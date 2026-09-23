Ratios use mean post-warmup times; summaries weight workloads equally.
Bandwidth follows scripts/utils.py. Parameter values are the recorded CSV fields.

| Metric | V100 (volta05) | L40 (ampere01) | A100 (ampere02) | H100 (hopper01) | RTX PRO 6000 Blackwell (bw01) |
|---|---|---|---|---|---|
| bits speedup vs best competitor: mean / min / max | 2.13× / 1.55× / 2.65× | 1.29× / 0.98× / 2.24× | 1.95× / 1.12× / 2.64× | 2.09× / 1.27× / 2.88× | 1.42× / 1.02× / 2.05× |
| bits peak-speedup workload | n=1048576, q=1024, d=1, k=512 | n=4194304, q=256, d=1, k=2048 | n=262144, q=4096, d=1, k=256 | n=1048576, q=1024, d=1, k=256 | n=4194304, q=256, d=1, k=1024 |
| bits minimum-speedup workload | n=262144, q=4096, d=1, k=32 | n=262144, q=4096, d=1, k=32 | n=262144, q=4096, d=1, k=2048 | n=16777216, q=64, d=1, k=32 | n=16777216, q=64, d=1, k=32 |
| bits speedup at q=64: mean | 1.97× | 1.29× | 1.55× | 1.78× | 1.40× |
| bits wins vs best competitor | 28/28 | 23/28 | 28/28 | 28/28 | 28/28 |
| bits throughput: peak (distances/s) | 2.1418e+11 | 2.0743e+11 | 4.0775e+11 | 4.7813e+11 | 3.7385e+11 |
| bits bandwidth utilization: mean / peak | 82.08% / 88.65% | 88.61% / 89.44% | 64.02% / 78.50% | 72.87% / 87.36% | 85.17% / 87.21% |
| Prefetch speedup gain: mean / min / max | 1.75% / -7.39% / 11.64% | -0.02% / -0.44% / 1.14% | 5.74% / -2.48% / 24.04% | 3.83% / -3.18% / 17.48% | 6.20% / -0.40% / 25.40% |
| Sort-in-registers speedup: mean / min / max | 2.86× / 1.86× / 10.28× | 2.43× / 1.41× / 8.89× | 2.71× / 1.54× / 10.16× | 3.00× / 1.66× / 11.99× | 3.09× / 1.67× / 10.90× |
| Sort-in-registers speedup: median | 2.18× | 1.81× | 2.11× | 2.10× | 2.33× |
| Sort-in-registers fastest | 28/28 | 26/28 | 28/28 | 25/28 | 25/28 |
| Fixed partial-bitonic: block; worst slowdown | 128; 45.52% | 128; 29.21% | 128; 33.80% | 128; 31.28% | 128; 39.29% |
| Fixed partial-bitonic-warp: block; worst slowdown | 128; 96.21% | 128; 44.16% | 128; 53.92% | 128; 29.25% | 128; 81.07% |
| Fixed partial-bitonic-warp-static: block; worst slowdown | 128; 83.58% | 128; 42.44% | 128; 57.99% | 128; 38.27% | 128; 74.14% |
| Fixed partial-bitonic-regs: block; worst slowdown | 128; 49.45% | 128; 30.37% | 128; 17.60% | 128; 21.49% | 128; 42.12% |
| Buffer speedup (ascending): mean / min / max | 5.48× / 2.61× / 7.00× | 5.21× / 3.50× / 7.52× | 6.00× / 4.91× / 7.04× | 4.96× / 3.78× / 6.19× | 3.83× / 2.00× / 6.74× |
| Buffer speedup (identity): mean / min / max | 5.24× / 2.52× / 6.77× | 5.30× / 3.63× / 7.55× | 5.72× / 3.96× / 7.00× | 4.77× / 3.47× / 6.12× | 3.76× / 1.87× / 6.81× |
| Buffer speedup (descending): mean / min / max | 0.77× / 0.57× / 0.88× | 0.72× / 0.48× / 0.88× | 0.81× / 0.59× / 0.92× | 0.79× / 0.61× / 0.92× | 0.69× / 0.39× / 0.87× |
| Fixed bits parameters: block; items; degree | 512; 5,1,1; 1 | 512; 5,1,1; 1 | 512; 7,1,1; 1 | 256; 13,1,1; 1 | 512; 4,1,1; 1 |
| Fixed bits worst slowdown | 16.43% | 1.72% | 27.11% | 31.60% | 6.64% |
| MAGMA-distance wins vs other plotted kernels | 25/28 | 24/28 | 24/28 | 27/28 | 22/28 |
| Distance parameters fixed per d: worst slowdown | 1.94% | 5.29% | 3.53% | 1.29% | 0.63% |
| bits-fused vs RAFT, d<=8: mean / min / max | 3.45× / 2.22× / 6.45× | 3.08× / 1.95× / 5.29× | 2.95× / 1.63× / 5.85× | 2.75× / 1.48× / 5.55× | 2.34× / 1.62× / 3.21× |
| bits-fused vs RAFT, d<=8: speedup >2× | 32/32 | 30/32 | 28/32 | 23/32 | 24/32 |
| bits-fused wins vs RAFT | 48/48 | 48/48 | 48/48 | 47/48 | 48/48 |
| bits-fused vs two-phase: mean / min / max | 2.73× / 1.47× / 4.06× | 5.45× / 2.88× / 8.62× | 1.90× / 0.82× / 3.22× | 1.78× / 0.89× / 2.86× | 3.05× / 1.49× / 5.00× |
| bits-fused wins vs two-phase | 72/72 | 72/72 | 66/72 | 65/72 | 72/72 |
| bits-fused wins vs two-phase, d=4 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| bits-fused wins vs two-phase, d=8 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| bits-fused wins vs two-phase, d=16 | 24/24 | 24/24 | 18/24 | 17/24 | 24/24 |
| RAFT throughput loss at k=64 vs k=8: mean / min / max | 48.36% / 40.45% / 58.14% | 39.17% / 37.86% / 40.59% | 46.47% / 42.91% / 53.74% | 47.77% / 44.09% / 54.74% | 14.35% / 10.60% / 19.03% |
| bits-fused throughput loss at k=256 vs k=8: mean / min / max | 24.92% / 14.41% / 31.43% | 13.03% / -7.28% / 25.42% | 11.08% / -2.92% / 28.28% | 6.86% / -2.90% / 15.94% | 6.43% / -8.74% / 24.28% |
| bits-fused wins vs bits + zero computation, d=4 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| bits-fused wins vs bits + zero computation, d=8 | 24/24 | 24/24 | 18/24 | 24/24 | 24/24 |
| bits-fused wins vs bits + zero computation, d=16 | 24/24 | 24/24 | 5/24 | 12/24 | 24/24 |
| Two-phase distance matrix (GiB) | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| Fixed bits-fused parameters: query tile; items tuple; degree | 1; 8,4,1; 2 | 2; 4,4,4; 1 | 2; 4,4,1; 4 | 1; 8,4,2; 1 | 2; 4,8,1; 4 |
| Fixed bits-fused worst slowdown | 34.34% | 45.62% | 41.57% | 27.10% | 33.04% |
| Fixed bits-fused vs RAFT: worst overhead (tuning/eval) | -14.30% | -6.43% | 22.40% | 6.38% | -2.48% |
| Fixed bits-fused vs RAFT: worst workload | n=131072, q=8192, d=16, k=8 | n=131072, q=8192, d=16, k=8 | n=1048576, q=1024, d=16, k=8 | n=1048576, q=1024, d=16, k=8 | n=131072, q=8192, d=16, k=8 |
| Fixed bits-fused wins vs two-phase (tuning/eval) | 72/72 | 72/72 | 60/72 | 60/72 | 72/72 |

| Run family | volta05 | ampere01 | ampere02 | hopper01 | bw01 |
|---|---|---|---|---|---|
| bitonic-sort | 108428 | 108429 | 108430 | 108431 | 35760909 |
| buffer | 108406, 108410, 108414 | 108404, 108409, 108415 | 108405, 108411, 108413 | 108407, 108408, 108412 | 35760910, 35760911, 35760912 |
| kselection | 108425 | 108426 | 108427 | 108424 | 35760913 |
| distances | 108421 | 108422 | 108420 | 108423 | 35760914 |
| fused | 108418 | 108419 | 108417 | 108416 | 35760915 |
| opt-bitonic-sort | 108290 | 108293 | 108291 | 101133 | 35835597 |
| opt-ipt | 108190, 108202, 108205 | 108187, 108199, 108203 | 108188, 108200, 108206 | 108189, 108201, 108204 | 35835590, 35835591, 35835592 |
| opt-distances | 108258 | 108256 | 108255 | 101154 | 35835600 |
| fused-cache-params | 108238, 108242, 108246 | 108241, 108245, 108248 | 108240, 108243, 108247 | 108239, 108244, 108249 | 35835594, 35835595, 35835596 |
