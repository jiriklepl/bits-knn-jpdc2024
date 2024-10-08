# kNN on GPUs

## Prerequisites

- `cmake` version at least 3.17
- `OpenMP`
- CUDA
- C++ compiler with C++23 support

## Compilation

Run the following commands in the root directory of this project to build the program and its dependencies. The initial compilation takes a long time (more than an hour in gpulab with 16 threads).

1. `git submodule update --init --recursive`
2. `cmake -E make_directory build-release`
3. `cmake -B build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -S .`
4. `cmake --build build-release --config Release -j16`

Use a different value for `CMAKE_CUDA_ARCHITECTURES` if you are not compiling the code for the Ampere architecture. This process creates an executable program `knn` and a unit test program `test`.

## Structure of the project

- `src/distance`: implementation of distance functions (mostly kernels for Euclidean distance)
- `src/topk/singlepass`: implementation of single-pass k-selection kernels
- `src/topk/multipass`: implementation of multi-pass k-selection kernels
- `doc/05-evaluation`: contains shell scripts for running all experiments (`run-*.sh`), and python3 scripts for plotting the results (`plot-*.py`). The python scripts require python 3, Matplotlib, and Pandas.

## Running the experiments

The compiled program (`./knn`) has several command line options:

```txt
  -a, --algorithm arg         Used algorithm (default: serial)
  -r, --repeat arg            Number of executions (default: 1)
  -k, --knn arg               Number of nearest neighbors (default: 1)
  -n, --number arg            Number of objects in the database (default:
                              1024)
  -q, --query arg             Number of query objects (default: 1024)
  -d, --dimension arg         Dimension of objects (default: 10)
      --seed arg              Seed for the random number generator
                              (default: 0)
      --block-size arg        Block size for CUDA kernels (default: 256)
      --items-per-thread arg  Number of items per thread in the fused-regs
                              kernel (two comma separated values) (default:
                              0)
  -v, --verify arg            Verify the results against selected
                              implementation (default: off)
      --point-layout arg      Layout of the point matrix - "column" for
                              column major or "row" for row major (default:
                              row)
      --query-layout arg      Layout of the query matrix - "column" for
                              column major or "row" for row major (default:
                              row)
      --deg arg               degree of parallelism used for single-query
                              problems (default: 1)
      --no-output             Do not copy the result back to CPU
      --random-distances      Use random unique distances instead of actual
                              distances.
  -h, --help                  Show help message
```

The following program runs all implemented unit tests:

```bash
./test
```

The following command prints a help message including the list of implemented kernels (available values for the `-a` option).

```bash
./knn --help
```

Some options are applicable to a subset of kernels only. This includes the `--block-size`, `--items-per-thread`, `--deg`, and `--no-output` options. Some kernels are only implemented for a subset of parameters (e.g., only for some values of `-k`).

## Experiments

Each experiment can be run using a bash script. All scripts are in the `doc/05-evaluation` folder. The scripts assume that the compiled executable is in `build-release/knn`. They can be run directly from the command line, but they also contain a SLURM configuration specifically for [KSI Clusters](https://gitlab.mff.cuni.cz/mff/hpc/clusters). Run the `sbatch run-<name>.sh` command to execute the script in [KSI Clusters](https://gitlab.mff.cuni.cz/mff/hpc/clusters).

**Prerequisites of the bash scripts** (you should ignore this if you use `sbatch` in gpulab):

- GPU with at least 32 GiB of global memory (i.e., partition volta04 or volta05 in gpulab).
- The results of `run-<name>.sh` are saved in `data/<name>.csv` (`sbatch` automatically stores the data in the correct folder).

### Plotting the results

All experiments also have a python3 script to plot the results. It requires python 3, Matplotlib, and Pandas to run. Run the following commands to install python dependencies in the `doc/05-evaluation` folder:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python3 -m pip install -r requirements.txt`

All bash scripts save the measurements in the `doc/05-evaluation/data` folder. This is done automatically if you use `sbatch`. Python scripts that plot the results read the measurements from this folder.

### Common distance functions

Comparison of Euclidean distance computation kernels (Figure 5.1 in the thesis). It compares a baseline kernel (one distance per thread) with a kernel proposed by Kuang et al. (which caches small matrix tiles in shared memory), a modified matrix multiplication kernel from the MAGMA library (which caches matrix tiles in shared memory and registers), and a matrix multiplication kernel from the cuBLAS library. The cuBLAS approach expresses the distances in terms of a dot product. The matrix multiplication kernel is used to compute the dot product.

For sufficient number of queries (at least 128) and dimension of vectors (at most 128), the modified MAGMA kernel outperformed the cuBLAS approach.

- *run*: `run-distances.sh` (~1 hour on Tesla V100)
  - output: `data/distances.csv`
- *plot*: `plot-distances.py`
  - output: `figures/distances.pdf` (Figure 5.1 in our thesis)
- *implementation*:
  - baseline: `src/distance/baseline_distance.{cu,hpp}` (option: `-a baseline-dist`)
  - Kuang et al.: `src/distance/tiled_distance.{cu,hpp}` (option: `-a tiled-dist`)
  - Modified MAGMA: `src/distance/abstract_gemm.cuh` (option: `-a magma-dist`)
  - cuBLAS: `src/distance/cublas_distance.{cu,hpp}` (option: `-a cublas-dist`)

### Optimizations: Bitonic sort

Evaluation of Bitonic sort optimizations using partial sorting kernels. The baseline kernel is a partial sorting top k selection which uses a naive Bitonic sort implementation in shared memory. For small strides, we implemented a version which uses warp shuffles. The final version in registers is described in detail in Section 4.1.1 in our thesis.

The optimized Bitonic sort achieved a consistent speed-up of about two for all tested values of k.

- *run*: `run-opt-bitonic-sort.sh` (~3 minutes on Tesla V100)
  - output: `data/opt-bitonic-sort.csv`
- *plot*: `plot-opt-bitonic-sort.py`
  - output: `figures/opt-bitonic-sort.pdf` (Figure 5.2 in our thesis)
- *implementation* (kernel implementation in `src/topk/singlepass/partial_bitonic.{cu,hpp}`):
  - sort in shared memory: `src/topk/singlepass/bitonic_sort.cuh` (`-a partial-bitonic`)
  - with warp shuffles: `src/topk/singlepass/bitonic_sort.cuh` (`-a partial-bitonic-warp`)
  - in registers: `src/topk/singlepass/bitonic_sort_regs.cuh` (`-a partial-bitonic-regs`)

### Optimizations: buffering

Evaluation of using a buffer in a partial sorting top k selection kernel. Values are first added to a buffer. When the buffer fills up with distances smaller than the kth smallest distance found so far, it is merged with the top k result.

Buffering significantly reduced the number of merge operations (Section 4.1.6 in our thesis). We saw a speed-up above four when compared with the optimized partial Bitonic sort kernel from the previous section.

- *run*: `run-opt-buffer.sh` (~3 minutes on Tesla V100)
  - output: `data/opt-buffer.csv`
- *plot*: `plot-opt-buffer.py`
  - output: `figures/opt-buffer.pdf` (Figure 5.3 in our thesis)
- *implementation*:
  - Partial Bitonic sort: `src/topk/singlepass/partial_bitonic.{cu,hpp}` (option `-a partial-bitonic-regs`)
  - with buffering (the bits kernel): `src/topk/singlepass/bits_kernel.cuh` (option `-a bits`)

### Optimizations: global memory throughput

We try to keep enough global memory transactions in flight by starting several read transactions per thread in each iteration. We also tried to insert prefetch instructions which load distances to the L2 cache. Multiple reads per thread significantly improve the performance of the kernel. We can achieve a speed-up above two for all tested values of k when compared with the same kernel with only one read per thread. In some instances, for larger values of k, we saw a speed-up above three.

- *run*: `run-opt-ipt.sh` (~10 minutes on Tesla V100)
  - output: `data/opt-ipt.csv`
- *plot*: `plot-opt-ipt.py`
  - output: `data/opt-ipt.pdf` (Figure 5.4 in our thesis)
- *implementation*: `src/topk/singlepass/bits_kernel.cuh` (option `-a bits` and `-a bits-prefetch`)

### Warp Select parameters

The default configuration of the Warp Select kernel from the [FAISS library](https://github.com/facebookresearch/faiss) does not perform as well as it could on Tesla V100. We wrote a script which runs the kernel with varying thread queue size and finds the best configuration for each value of k. The plot shows that the Warp Select kernel with alternative thread queue size performs on par with or better than the default configuration.

- *run*: `run-params-warp-select.sh` (~14 minutes on Tesla V100)
  - output: `data/params-warp-select.csv`
- *plot*: `plot-params-warp-select.py`
  - output: `data/params-warp-select.pdf` (Figure 5.5 in our thesis)
- *implementation*: in the [FAISS library](https://github.com/facebookresearch/faiss).

### Single-pass k-selection

Evaluation of single-pass, small k selection methods. We show implementation of our optimization Bitonic select (bits) kernel, the Warp Select and Block Select kernels from the [FAISS library](https://github.com/facebookresearch/faiss),and the Merge queue kernel from the fgknn library. Our implementation of k selection outperforms other k selection kernels in all tested configurations. The kernel approaches 80% of peak theoretical throughput with a typical configuration (large database, a relatively small number of queries, and small k <= 128).

- *run*: `run-kselection-sp.sh` (~14 minutes on Tesla V100)
  - output: `data/kselection-sp.csv`
- *plot*: `plot-kselection-sp.py`
  - output: `data/kselection-sp.pdf` (Figure 5.6 in our thesis)
- *implementation*:
  - Warp Select, Block Select: in the [FAISS library](https://github.com/facebookresearch/faiss) (option: `-a warp-select`, `-a warp-select-tuned`, `-a block-select`, `-a block-select-tuned`).
  - Merge queue: in the fgknn library (`src/fgknn`, option: `-a fgknn-buffered`)
  - bits: `src/topk/singlepass/bits_kernel.cuh` (option: `-a bits`)

### Single-query selection

Evaluation of the single-query modification of the bits kernel on a large database with different number of partitions. It shows that the kernel achieves a peak performance with 256 partitions for small k <= 512. Larger values of k require fewer partitions.

- *run*: `run-bits-sq.sh` (~20 minutes on Tesla V100)
  - output: `data/bits-sq.csv`
- *plot*: `plot-bits-sq.py`
  - output: `data/bits-sq.pdf` (Figure 5.7 in our thesis)
- *implementation*: `src/topk/singlepass/bits_kernel.cuh`, `src/topk/singlepass/bits_knn.{cu,hpp}` (option: `-a bits-sq`)

### Fused kernel

Comparison of the fused kNN kernel with the bits kernel and the modified MAGMA kernel for distance computation. The fused kernel was faster for small dimensions. The second plot shows that the fused kernel requires more queries to be effective than the bits kernel (it achieved a peak throughput with at least eight thousand queries).

- *run*: `run-fused.sh` (~14 minutes on Tesla V100)
  - output: `data/fused.csv`
- *plot 1*: `plot-fused.py`
  - output: `data/fused.pdf` (Figure 5.8 in our thesis)
- *plot 2*: `plot-fused.py`
  - output: `data/fused.pdf` (Figure 5.9 in our thesis)
- *implementation*: `src/topk/singlepass/fused_kernel.cuh`, `src/topk/singlepass/fused_knn.{cu,hpp}` (option: `-a fused-regs`)

### Multi-pass k-selection

Evaluation of multi-pass, large k selection algorithms. The Radix Select kernel outperformed the Sample Select on the smallest tested database with only 32 thousand vectors. For larger databases, the Sample Select kernel performed better.

- *run*: `run-kselection-mp.sh` (~40 minutes on Tesla V100)
  - output: `data/kselection-mp.csv`
- *plot*: `plot-kselection-mp.py`
  - output: `data/kselection-mp.pdf` (Figure 5.11 in our thesis)
- *implementation*:
  - baseline CUB sort in multiple CUDA streams (option `-a cub-sort`)
  - baseline segmented CUB sort (option `-a cub-sort-seg`)

### Final kernel

Comparison of the best multi-query kernels for different configurations of kNN. This evaluation includes both parts of kNN -- distance computation and k selection.

- *run*: `run-eval-mq.sh` (~1 hour and 20 minutes on Tesla V100)
  - output: `data/eval-mq.csv`
- *plot*: `plot-eval.py`
  - output: `data/eval-mq.pdf` (Figure 5.12 in our thesis)
- *implementation*:
  - bits: `src/topk/singlepass/bits_kernel.cuh` (option: `-a bits`)
  - bits in global memory: `src/topk/multipass/bits_global.{cu,hpp}` (option: `-a bits-global`)
  - fused kernel: `src/topk/singlepass/fused_kernel.cuh` (option: `-a fused-regs`)

### Comparison with a CPU implementation

Comparison of a parallel CPU kNN implementation using the Eigen library for distance computation and OpenMP with our GPU implementation of the bits kernel. kNN computation is significantly faster on a GPU. We achieved a speed-up above 80 with a typical configuration. The data transfers are faster than computation for small dimensions (<= 16). However, if we process more than one batch of queries, the matrix of database vectors does not have to be transferred to GPU every time.

- *run*: `run-parallel.sh` (~1 hour on Tesla V100 and Intel Xeon Gold 5218)
- *plot*:
  - GPU speed-up: `plot-gpu-speedup.py` (Figure 5.13 in our thesis)
  - memory transfers: `plot-gpu-transfers.py` (Figure 5.14 in our thesis)
- *implementation*:
  - GPU (the bits kernel): `src/topk/singlepass/bits_kernel.cuh` (option: `-a bits`)
  - CPU: `src/topk/singlepass/parallel_knn.{cpp,hpp}` (option: `-a parallel`)

## Overview of implemented algorithms

### Single-pass, multi-query kernels

- `serial`: serial implementation on CPU.
- `parallel`: parallel implementation using Eigen and OpenMP on CPU.
- `partial-bitonic`, `partial-bitonic-warp`, `partial-bitonic-regs`: partial sorting k-selection kernel using Bitonic sort with different Bitonic sort implementations (naive sort in shared memory, sort using warp shuffles, and our optimized sort in registers).
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `bits`, `bit-prefetch`: our Bitonic Select kernel (multi-query, small k-selection kernel) without prefetching and with prefetching.
  - `--items-per-thread arg` controls the number of reads per thread (allowed range `[1, 16]`).
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `warp-select`: WarpSelect kernel from the FAISS library with the default configuration.
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `block-select`: BlockSelect kernel from the FAISS library with the default configuration.
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `warp-select-tuned`: WarpSelect kernel from the FAISS library with tuned parameters for Tesla V100.
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `block-select-tuned`: BlockSelect kernel from the FAISS library with tuned parameters for Tesla V100.
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.
- `fused-regs`: kernel, which fuses Euclidean distance computation and k-selection.
  - `-k` is limited to 4, 8, 16, 32, 64, 128.

### Multi-pass kernels

- `bits-global`: adaptation of our single-pass, small k-selection kernel, which stores the top k result in global memory.
- `cub-sort`: Radix sort from the CUB library, which calls multiple CUB sort kernels in parallel.
- `cub-sort-seg`: segmented Radix sort implementation from the CUB library.
  - the `postprocessing` phase sorts the result using the segmented CUB sort.
  - the `postprocessing` phase sorts the result using the segmented CUB sort.
- `bits-sq`: our single-query Bitonic Select kernel.
  - `--deg arg` controls the number of partitions (the degree of parallelism)
  - `--items-per-thread arg` controls the number of reads per thread (allowed range `[1, 16]`).
  - `-k` is limited to 32, 64, 128, 256, 512, 1024, and 2048.

### Distance kernels

- `baseline-dist`: baseline implementation in which each thread computes one Euclidean distance between a query vector and a database vector.
- `tiled-dist`: our implementation of the kernel due to Kuang et al.
  - `--block-size` should be `8`, `16`, or `32` (the kernel uses square thread block tiles so the total number of threads will be `64`, `256`, and `1024`, respectively).
- `cudlass-dist`: Euclidean distance computation using modified matrix multiplication from the cutlass library.
- `cublas-dist`: Euclidean distance computation using matrix multiplication from the cuBLAS library.
- `magma-dist`: Euclidean distance computation using modified matrix multiplication from the MAGMA library.
- `magma-kl-dist`: example of a more complex distance function (KL divergence) using modified matrix multiplication from the MAGMA library.

## Output format

The `./knn` program prints the data in a CSV format. The first line always contains the header which comprises of the following columns.

- *algorithm*: name of the kernel.
- *iteration*: number of the iteration (exactly `--repeat` iterations are executed).
- *point_count*: number of database vectors.
- *query_count*: number of queries.
- *dim*: dimension of database and query vectors.
- *block_size*: value of `--block-size` from the command line. This may not reflect the actual thread block size if the selected algorithm ignores the `--block-size` argument.
- *k*: number of nearest neighbors returned for each query.
- *items_per_thread*: three integers separated by a comma. Meaning depends on the selected kernel.
- *phase*: phase of the execution.
  - *prepare*: initialization phase.
  - *distances*: distance computation kernel (excluding data transfers, memory allocation, and initialization).
  - *selection*: top k selection kernel (excluding data transfers, memory allocation, and initialization).
  - *postprocessing*: processing of the top k result (e.g., sorting if the selected kernel returns an unsorted result).
  - *finish*: transfer of the result to the main memory of CPU.
  - *transfer-in*: data transfers from CPU memory to GPU memory.
  - *transfer-out*: data transfer from GPU memory to CPU memory.
- *time*: duration of the phase in seconds.

The program also prints a checksum to the error stream, which is the sum of the top k labels of the first query (mod `2^32`) for quick verification of the correctness of the results. This should not be used for rigorous verification as the top k output is ambiguous if there are some objects with the same distance.
