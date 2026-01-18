# Algorithms

This document describes the proposed algorithms and the other evaluated algorithms used in the benchmarks. The proposed algorithms `bits` and `bits-fused` can be used by including their respective headers located in the `include/bits` directory without further dependencies (except the CUDA toolkit).


## Proposed algorithms

The benchmarking binary `knn` described in [knn.md](knn.md) can run the following proposed algorithms for top-k selection and k-NN search (fused top-k selection with L2 distance computation):

- The proposed `bits` algorithm for top-k selection
  - In the benchmarking binary `knn`, the `bits` algorithm with prefetching enabled is run with the `-a bits-prefetch` argument
    - `-a bits` runs the algorithm without prefetching; for the tested problems, their performance is quite similar
  - Implemented in the  `bits_kernel` kernel in [include/bits/topk/singlepass/detail/bits_kernel.cuh](../include/bits/topk/singlepass/detail/bits_kernel.cuh), line 320
  - In the visualized benchmarks, we always enable prefetching, and we set block size and batch size (items per thread) parameters based on the [tuning experiments](../README.md#optional-tuning-the-algorithm-parameters).

- The optimized partial Bitonic sort algorithm variants `baseline`, `warp-shuffle`, and `sort-in-registers` performing top-k selection
  - In the benchmarking binary `knn`, the algorithms are run with the `-a partial-bitonic`, `-a partial-bitonic-warp-static`, and `-a partial-bitonic-regs` arguments, respectively
  - All implemented in [src/topk/singlepass/partial_bitonic.cu](../src/topk/singlepass/partial_bitonic.cu)

- The `bits-fused` algorithm for k-NN search (fused top-k selection with L2 distance computation)
  - In the benchmarking binary `knn`, the proposed fused algorithm is run with the `-a fused-cache` argument
  - Implemented in the `fused_cache_kernel` struct in [include/bits/topk/singlepass/fused_cache_kernel_structs.cuh](../include/bits/topk/singlepass/fused_cache_kernel_structs.cuh), line 666
  - Entry point is the `launch_fused_cache` kernel in [include/bits/topk/singlepass/detail/fused_cache_kernel.cuh](../include/bits/topk/singlepass/detail/fused_cache_kernel.cuh), line 27
  - In the visualized benchmarks, we always set its parameters based on the [tuning experiments](../README.md#optional-tuning-the-algorithm-parameters).

The implementations of the proposed algorithms in the `include/bits` directory can be included in the user's codebase.

## Other evaluated algorithms

Here is a list of the evaluated algorithms used in the benchmarks other than the proposed algorithms `bits` and `bits-fused`, and the optimized partial Bitonic sort algorithms.

State-of-the-art algorithms for top-k selection evaluated in the paper:

- `BlockSelect` from the FAISS library; proposed by Johnson et al. (2012)
  - called in [include/bits/topk/singlepass/detail/warp_select.cuh](../include/bits/topk/singlepass/detail/warp_select.cuh)'
  - in the benchmarking binary `knn`, the algorithm is run with the `-a block-select-tunable` argument (we tune the block size and the thread queues parameters)
- `AIR-topk` (radix select_k) from the RAFT library; proposed by Zhang et al. (2023)
  - called in [src/topk/multipass/air_topk.cu](../src/topk/multipass/air_topk.cu)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a air-topk` argument
- `GridSelect` proposed by Zhang et al. (2023)
  - called in [src/topk/singlepass/grid_select.cu](../src/topk/singlepass/grid_select.cu)
  - it is the only evaluated algorithm with no available source code; we use the precompiled library available at [https://github.com/ZhangJingrong/gpu_topK_benchmark/blob/master/third_party/libgridselect.so](https://github.com/ZhangJingrong/gpu_topK_benchmark/blob/master/third_party/libgridselect.so)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a grid-select` argument

Distance computation algorithms evaluated in the paper (state-of-the-art and a naive baseline):

- `baseline`: a naive implementation of the distance computation; each distance computed by a separate thread
  - called in [src/distance/baseline_distance.cu](../src/distance/baseline_distance.cu)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a baseline-dist` argument
- `cuBLAS`: a distance computation accelerated by the cuBLAS library
  - called in [src/distance/cublas_distance.cu](../src/distance/cublas_distance.cu)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a cublas-dist` argument
- The modified MAGMA GEMM kernel for distance computation used by bf-knn; proposed by Li et al. (2015)
  - called in [src/distance/magma_distance.cu](../src/distance/magma_distance.cu)
  - we consider a variant that expands the distance computation formula into $||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y$ and leaves out the query vector norm as it does not affect the top-k selection
    - reimplemented in [include/bits/distance/abstract_gemm.cuh](../include/bits/distance/abstract_gemm.cuh); this implementation roughly corresponds to the kernel in [https://github.com/geomlab-ucd/bf-knn/blob/master/bf_knn_device.cu](https://github.com/geomlab-ucd/bf-knn/blob/master/bf_knn_device.cu)
  - in the paper and plots, referred to as `MAGMA-distance`
  - in the benchmarking binary `knn`, the algorithm is run with the `-a magma-part-dist` argument

Top-k selection algorithms that are not shown in the paper (superseded by the previously mentioned algorithms):

- `RadiK` proposed by Li et al. (2024)
  - called in [src/topk/multipass/radik_knn.cu](../src/topk/multipass/radik_knn.cu)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a radik` argument
- `WarpSelect` from the FAISS library, proposed by Johnson et al. (2012)
  - called in [include/bits/topk/singlepass/detail/warp_select.cuh](../include/bits/topk/singlepass/detail/warp_select.cuh)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a warp-select-tunable` argument (we tune the block size and the thread queues parameters)
- `warpsort` from the RAFT library, based on the `WarpSelect` algorithm
  - called in [src/topk/singlepass/raft_warpsort.cu](../src/topk/singlepass/raft_warpsort.cu)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a warpsort` argument

Distance computation algorithms that are not shown in the paper (superseded by the previously mentioned algorithms):

- `tiled-distance` proposed by Kuang et al. (2009)
  - implemented in [src/distance/tiled_distance.cu](../src/distance/tiled_distance.cu)
  - in plots, referred to as `Kuang et al.`
  - in the benchmarking binary `knn`, the algorithm is run with the `-a tiled-dist` argument
- The modified MAGMA GEMM for distance computation used by bf-knn; proposed by Li et al. (2015)
  - variant referred to as `MAGMA-distance (unexpanded)`
  - called in [src/distance/magma_distance.cu](../src/distance/magma_distance.cu)
  - this variant uses the distance computation formula in unexpanded form (computes the sum of the squared differences of the coordinates)
  - in the benchmarking binary `knn`, the algorithm is run with the `-a magma-dist` argument
