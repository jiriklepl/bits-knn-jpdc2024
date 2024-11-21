# Artifact for GPU-accelerated Parallel K-nearest Neighbors and Top-k Selection for Multi-query Workloads

This is the replication package for the paper *GPU-accelerated Parallel K-nearest Neighbors and Top-k Selection for Multi-query Workloads* submitted to the [Journal of Parallel and Distributed Computing](https://www.sciencedirect.com/journal/journal-of-parallel-and-distributed-computing) (JPDC).

The replication package contains the following artifacts:

- The source code for the benchmarking of the algorithms in directories [./src/](./src/) and [./include/](./include/)
  - Contains the proposed algorithms `bits` for top-k selection and `bits-fused` for top-k selection with distance computation (k-NN search)
    - The `bits` algorithm refers to the `bits_kernel` kernel in [include/bits/topk/singlepass/detail/bits_kernel.cuh](include/bits/topk/singlepass/detail/bits_kernel.cuh), line 320

      In the visualized benchmarks, we always set the `PREFETCH` parameter to `true`; `BLOCK_SIZE` and `BATCH_SIZE` (items per thread) are chosen empirically based on the [tuning experiments](#optional-tuning-the-algorithm-parameters)

    - The `bits-fused` algorithm refers to the `fused_cache_kernel` struct in [include/bits/topk/singlepass/fused_cache_kernel_structs.cuh](include/bits/topk/singlepass/fused_cache_kernel_structs.cuh), line 666

        The `launch_fused_cache` kernel entry point for this algorithm is in [include/bits/topk/singlepass/detail/fused_cache_kernel.cuh](include/bits/topk/singlepass/detail/fused_cache_kernel.cuh), line 27

        Its parameters are chosen empirically based on the [tuning experiments](#optional-tuning-the-algorithm-parameters)

  - Contains the proposed optimized partial Bitonic sort algorithm variants `baseline`, `warp-shuffle` and `sort-in-registers` performing top-k selection
    - All implemented in [src/topk/singlepass/partial_bitonic.cu](src/topk/singlepass/partial_bitonic.cu)
- Scripts to run the benchmarks and produce the plots presented in the paper in the [./scripts/](./scripts/) directory
- The data collected from the benchmarks on the following hardware in the [./data/](./data/) directory:
  - NVIDIA Tesla V100 (Volta) - files named `*-volta05-*.csv`
  - NVIDIA A100 (Ampere) - files named `*-ampere02-*.csv`
  - NVIDIA H100 (Hopper) - files named `*-hopper01-*.csv`
  - NVIDIA L40 (Lovelace) GPUs - files named `*-ampere01-*.csv`
- The plots generated from the collected data in [./plots/](./plots/) directory

## Results reproduction

### Requirements

The following software and hardware is required to reproduce the results (the versions are the ones used in the original experiments, not strict requirements):

- `cmake` version 3.27.6
- CUDA Toolkit version 12.6 (`nvcc` compiler)
  - and NVIDIA GPU with compute capability 7.0, 8.0, 8.9 or 9.0 (V100, A100, L40 or H100)
- CUDA-compatible C++ compiler, preferably `gcc` version 13.2.0
- `git` version 2.43.5
- `python` version 3.9.18 (for visualization)

### Building the source code

To build the source code automatically, run the following commands (replace `CUDA_ARCHITECTURES` with the compute capability of your GPU, e.g., `90` for Hopper, and `NAME` with the chosen name that distinguishes your build from others):

```bash
# Clone the repository and update the submodules:
git clone https://github.com/jiriklepl/bits-knn-jpdc2024.git
cd bits-knn-jpdc2024
git submodule update --init --recursive

# Build the source code in the ./build-NAME directory:
./local-build.sh NAME CUDA_ARCHITECTURES build

# to build the `knn-minimal` version and the `test` target:
./local-build.sh NAME CUDA_ARCHITECTURES build-minimal
```

To build the source code manually, follow these steps:

0. Choose a `NAME` that will be used to distinguish your build from others (e.g., `build-volta`)

1. Clone the repository and update the submodules:

    ```bash
    git clone https://github.com/jiriklepl/bits-knn-jpdc2024.git
    cd bits-knn-jpdc2024
    git submodule update --init --recursive
    ```

2. Configure the build with `cmake` (replace `NAME` with the chosen name):

    ```bash
    cmake -B build-NAME -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="90" -S .
    ```

    (Replace the `CMAKE_CUDA_ARCHITECTURES` with the compute capability of your GPU, add `-DCMAKE_CXX_COMPILER=g++-13` to explicitly set the compiler)

3. Build the source code (replace `NAME` with the chosen name):

    ```bash
    cmake --build build-NAME --parallel 16
    ```

    This produces the `build/knn` executable that can be used to run the benchmarks.

    (Replace `--parallel 16` with the number of CPU cores available or reduce the number to avoid overloading the system)

    If you want to build the `knn-minimal` version or just the `test` target, run one of the following commands (replace `NAME` with the chosen name):

    ```bash
    cmake --build build-NAME --target knn-minimal
    cmake --build build-NAME --target test
    ```

### Running the benchmarks

The `local-build.sh` script is used to simplify redirection of the output to separate files `./data/EXPERIMENT-NAME-TIMESTAMP.csv` and `./data/EXPERIMENT-NAME-TIMESTAMP.err` (and also to build and test the source code).

In our laboratory `gpulab`, we use Slurm to run the benchmarks on various machines equipped with the NVIDIA GPUs. For these machines, we use equivalent scripts:

- `volta-build.sh` for the NVIDIA Tesla V100 (Volta) GPU-equipped machine
- `ampere-build.sh` for the NVIDIA A100 (Ampere) GPU-equipped machine
- `hopper-build.sh` for the NVIDIA H100 (Hopper) GPU-equipped machine
- `lovelace-build.sh` for the NVIDIA L40 (Lovelace) GPU-equipped machine

All these scripts follow the same pattern as the `local-build.sh` script in the following examples.

#### (Optional) Tuning the algorithm parameters

To tune the algorithm parameters, run the following commands (replace `CUDA_ARCHITECTURES` with the compute capability of your GPU, e.g., `90` for Hopper, and `NAME` with the chosen name):

```bash
# to benchmark various values for the items-per-thread parameter of the bits algorithm
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-opt-ipt.sh

# to benchmark various parameters of the variants of the `bits-fused` algorithm
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-fused-cache-params.sh
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-fused-params.sh

# to benchmark various parameters of the optimized Bitonic sort algorithms
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-opt-bitonic-sort.sh

# to benchmark various parameters of WarpSelect and BlockSelect algorithms from FAISS
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-params-warp-select.sh
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-params-block-select.sh

# to benchmark various parameters for various distance computation algorithms
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-opt-distances.sh
```

(this generates various `./data/EXPERIMENT-NAME-TIMESTAMP.csv` and `./data/EXPERIMENT-NAME-TIMESTAMP.err` files; the `.err` files contain checksums for the generated data in the `.csv` files and report incorrect parameters and other issues)

Then, run the following commands to generate `./scripts/optima.csv` and `./scripts/optima-dist.csv` files, run the following commands:

```bash
# produce the `./scripts/optima.csv` file:
./scripts/plot-all.sh analyze-opt

# produce the `./scripts/optima-dist.csv` file:
./scripts/plot-all.sh analyze-dist
```

#### Running the benchmarks

To run the benchmarks, run the following commands (assuming, you have already built and tuned the algorithms):

```bash
# to run the benchmarks for the optimized Bitonic sort algorithms:
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-bitonic-sort.sh

# to run the benchmarks for comparison of the best-performing Bitonic sort algorithm to the proposed bits algorithm (that adds buffering of filtered items):
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-buffer.sh uniform
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-buffer.sh uniform-ascending
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-buffer.sh uniform-descending

# to run the benchmarks comparing the proposed bits algorithm to State-of-the-Art algorithms for top-k selection:
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-kselection.sh

# to run the benchmarks for distance computation algorithms:
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-distances.sh

# to run the benchmarks for the bits-fused algorithm comparing it to the bits algorithm and the best-performing distance computation algorithm (MAGMA-distance):
./local-build.sh NAME CUDA_ARCHITECTURES ./scripts/run-fused.sh
```

Each command runs the given script and redirects its output to the `./data/EXPERIMENT-NAME-TIMESTAMP.csv` and `./data/EXPERIMENT-NAME-TIMESTAMP.err` files. The `.err` files contain checksums for the generated data in the `.csv` files and report incorrect parameters and other issues.

All the `.csv` files follow the same format:

```csv
algorithm,generator,preprocessor,iteration,point_count,query_count,dim,block_size,k,items_per_thread,deg,phase,time
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,prepare,7.8199999999999999e-07
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,transfer-in,3.0388478189706802e-03
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,distances,4.2674490000000004e-03
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,selection,1.5226899480000000e+00
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,postprocessing,5.6100000000000001e-07
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,finish,1.1387300000000000e-04
partial-bitonic,uniform,identity,0,16777216,64,1,64,32,"1,1,1",1,transfer-out,5.7119999837595969e-05
```

- `algorithm` is the name of the algorithm
- `generator` is the name of the data generator: in all cases, it is `uniform`
- `preprocessor` is the name of the data preprocessor
  - `identity` for no preprocessing (random order)
  - `ascending` for ascending order; all queries are zeroed and the database values are sorted in ascending order
  - `descending` for descending order; all queries are zeroed and the database values are sorted in descending order
- `iteration` is the iteration number; for the visualized benchmarks, 0-9 are warm-up iterations, 10-19 are measured iterations
- `point_count` is the number of points in the database
- `query_count` is the number of queries
- `dim` is the dimensionality of the points
- `block_size` is the block size used in the algorithm
  - in the `bits-fused` algorithm, this parameter denotes the first free parameter for the tiling used by the `MAGMA` kernel
- `k` is the number of nearest neighbors to find
- `items_per_thread` are up to three further parameters of the algorithm
  - in the `bits` algorithm, only the first parameter is used, and it denotes the number of items loaded by each thread in each batch of items
  - in the `bits-fused` algorithm, the parameters denote the remaining three free parameters of the tiling used by the `MAGMA` kernel; the fourth parameter is then computed so the resulting number of threads per block is 256
- if `deg` is greater than 1, the `bits-sq` variant of the `bits` algorithm splits the input items into `deg` groups and processes them separately, then merges the results; it is not used in the visualized benchmarks
- `phase` is the name of the phase of the algorithm, the following two phases are relevant to the experiments:
  - `distances` is the computation of the distances (used when evaluating the distance computation algorithms; when evaluating the fused distance computation and top-k selection algorithms, it is used to compare the combined time of the distance computation and the selection phase of the un-fused algorithms to the time of the fused algorithm)
  - `selection` is the top-k selection phase (used when evaluating the top-k selection algorithms; which includes all the experiments with the exception of the distance computation algorithms; fused algorithms report their time as `selection` time)
- `time` is the time in seconds spent in the given phase

### Plotting the results

To plot the results, run the following commands:

```bash
# (Figure 8 in the paper)
# to plot the results for the optimized Bitonic sort algorithms
./scripts/plot-all.sh bitonic-sort

# The same script plots the results for the comparison of the best-performing Bitonic sort algorithm to the proposed bits algorithm:

# (Figure 9 in the paper)
# to merges the results from the uniform, uniform-ascending, and uniform-descending experiments:
./scripts/plot-all.sh multibuffer

# (Figure 10 in the paper)
# to plot the results for the proposed bits algorithm compared to State-of-the-Art algorithms for top-k selection:
./scripts/plot-all.sh kselection > data/kselection-stats.csv

# (Figure 11 in the paper)
# to plot the results for the distance computation algorithms:
./scripts/plot-all.sh distances

# (Figure 12 in the paper)
# to plot the results for the bits-fused algorithm (same as the bits algorithm)
./scripts/plot-all.sh kselection
```

The plots are stored in the `./plots/` directory; names of the plots correspond to the names of the data files in the `./data/` directory (they share the same `EXPERIMENT-NAME-TIMESTAMP`). The only exception to this rule are the plots produced by the `multibuffer` command, which are stored in the `./plots/multibuffer-NAME.pdf` files as it merges all results sharing the same `NAME` prefix.

Some scripts also produce `./plots/extra-EXPERIMENT-NAME-TIMESTAMP.pdf` that do not filter out algorithms that have not been chosen for the visualization in the paper (as they were superseded by other algorithms).

The figures in the paper show the following plots:

- Figure 8: [`./plots/bitonic-sort-hopper01-108431.pdf`](./plots/bitonic-sort-hopper01-108431.pdf)
- Figure 9: [`./plots/multibuffer-hopper01.pdf`](./plots/multibuffer-hopper01.pdf)
- Figure 10: [`./plots/kselection-hopper01-108424-Uniform.pdf`](./plots/kselection-hopper01-108424-Uniform.pdf)
- Figure 11: [`./plots/distances-hopper01-108423.pdf`](./plots/distances-hopper01-108423.pdf)
- Figure 12: [`./plots/fused-hopper01-108416-Uniform.pdf`](./plots/fused-hopper01-108416-Uniform.pdf)

## License

The source code and the replication package are licensed under the MIT License. The license text is available in the `LICENSE.txt` file.

Some parts of the source code come from third-party projects. Their licenses are available in the `licenses.txt` file.

## Considered algorithms

Here is a list of the considered algorithms used in the benchmarks other than the proposed algorithms `bits` and `bits-fused`, and the optimized partial Bitonic sort algorithms.

State-of-the-Art algorithms for top-k selection considered in the paper:

- `BlockSelect` from the FAISS library; proposed by Johnson et al. (2012)
  - called in [include/bits/topk/singlepass/detail/warp_select.cuh](include/bits/topk/singlepass/detail/warp_select.cuh)
- `AIR-topk` (radix select_k) from the RAFT library; proposed by Zhang et al. (2023)
  - called in [src/topk/multipass/air_topk.cu](src/topk/multipass/air_topk.cu)
- `GridSelect` proposed by Zhang et al. (2023)
  - called in [src/topk/singlepass/grid_select.cu](src/topk/singlepass/grid_select.cu)
  - it is the only considered algorithm with no available source code; we use the precompiled library available at [https://github.com/ZhangJingrong/gpu_topK_benchmark/blob/master/third_party/libgridselect.so](https://github.com/ZhangJingrong/gpu_topK_benchmark/blob/master/third_party/libgridselect.so)

Distance computation algorithms considered in the paper (state-of-the-art and a naive baseline):

- `baseline`: a naive implementation of the distance computation; each distance computed by a separate thread
  - called in [src/distance/baseline_distance.cu](src/distance/baseline_distance.cu)
- `cuBLAS`: a distance computation accelerated by the cuBLAS library
  - called in [src/distance/cublas_distance.cu](src/distance/cublas_distance.cu)
- The modified MAGMA GEMM kernel for distance computation used by bf-knn; proposed by Li et al. (2015)
  - called in [src/distance/magma_distance.cu](src/distance/magma_distance.cu)
  - we consider a variant that expands the distance computation formula into $||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y$ and leaves out the query vector norm as it does not affect the top-k selection
    - reimplemented in [include/bits/distance/abstract_gemm.cuh](include/bits/distance/abstract_gemm.cuh)
  - in the paper and plots, referred to as `MAGMA-distance`

Fused distance computation and top-k selection algorithms considered in the paper:

- `fusedL2Knn` from the RAFT library; based on the `WarpSelect` algorithm
  - called in [src/topk/singlepass/rapidsai_fused.cu](src/topk/singlepass/rapidsai_fused.cu)
  - in the paper and plots, referred to as `raftL2-fused`

Top-k selection algorithms that are not shown in the paper as they were superseded by the previously mentioned algorithms:

- `RadiK` proposed by Li et al. (2024)
  - called in [src/topk/multipass/radik_knn.cu](src/topk/multipass/radik_knn.cu)
- `WarpSelect` from the FAISS library; proposed by Johnson et al. (2012)
  - called in [include/bits/topk/singlepass/detail/warp_select.cuh](include/bits/topk/singlepass/detail/warp_select.cuh)
- `warpsort` from the RAFT library; based on the `WarpSelect` algorithm
  - called in [src/topk/singlepass/raft_warpsort.cu](src/topk/singlepass/raft_warpsort.cu)

Distance computation algorithms that are not shown in the paper as they were superseded by the previously mentioned algorithms:

- `tiled-distance` proposed by Kuang et al. (2009)
  - implemented in [src/distance/tiled_distance.cu](src/distance/tiled_distance.cu)
  - in plots, referred to as `Kuang et al.`
- The modified MAGMA GEMM for distance computation used by bf-knn; proposed by Li et al. (2015)
  - variant reffered to as `MAGMA-distance (unexpanded)`
  - called in [src/distance/magma_distance.cu](src/distance/magma_distance.cu)
  - this variant uses the distance computation formula in unexpanded form (computes the sum of the squared differences of the coordinates)
