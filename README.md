# Replication package for the paper *Towards Optimal GPU-accelerated K-Nearest Neighbors Search*

[![DOI](https://zenodo.org/badge/869231733.svg)](https://doi.org/10.5281/zenodo.14212691)

This is the replication package for the paper *Towards Optimal GPU-accelerated K-Nearest Neighbors Search* submitted to the [Journal of Parallel and Distributed Computing](https://www.sciencedirect.com/journal/journal-of-parallel-and-distributed-computing) (JPDC).

The paper presents a study of state-of-the-art algorithms for top-k selection and distance computation (used in k-NN search) on (NVIDIA) GPUs. It proposes two novel algorithms, `bits` and `bits-fused`, outperforming the state-of-the-art algorithms. The proposed algorithms are based on the partial Bitonic sort algorithm by Kruliš et al. (2015) and are enhanced with novel optimizations. All algorithms are implemented in CUDA ([CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)) and evaluated on NVIDIA GPUs (Tesla V100, A100, H100, and L40).

To reproduce the results, follow the instructions in [Results reproduction](#results-reproduction). For a quick demonstration of the proposed algorithms and tests validating their correctness, follow the instructions in [Demonstration and testing](#demonstration-and-testing).

Section [Plotting the results](#plotting-the-results) describes how to visualize the results and maps the produced plots to the figures in the paper. It also discusses the reported speedup over the state-of-the-art algorithms and the relative throughput as a fraction of the theoretical throughput limit deduced from the memory bandwidth of the given GPU.

The replication package contains the following artifacts:

1. The source code is in directories [src/](src/) and [include/](include/). It contains the code for the benchmarking binary `knn`, which runs benchmarks for the proposed algorithms and the evaluated state-of-the-art algorithms. The algorithms and their source files are listed in the [algorithms.md](docs/algorithms.md).
2. Scripts to run the benchmarks and produce the plots presented in the paper are in the [scripts/](scripts/) directory
3. The data collected from the benchmarks on the following hardware are in the [data/](data/) directory:
    - NVIDIA Tesla V100 (Volta) - files named `*-volta05-*.csv`
    - NVIDIA A100 (Ampere) - files named `*-ampere02-*.csv`
    - NVIDIA H100 (Hopper) - files named `*-hopper01-*.csv`
    - NVIDIA L40 (Lovelace) GPUs - files named `*-ampere01-*.csv`
4. The plots generated from the collected data are in [plots/](plots/) directory


## Results reproduction


### Requirements

The following software and hardware are required to reproduce the results (the versions are the ones used in the original experiments, not strict requirements):

- `cmake` version 3.27.6
- CUDA Toolkit version 12.6 (`nvcc` compiler)
  - and NVIDIA GPU with compute capability 7.0, 8.0, 8.9, or 9.0 (Tesla V100, A100, L40, or H100) and with at least 32 GiB of memory
- CUDA-compatible C++ compiler, preferably `gcc` version 13.2.0
- `git` version 2.43.5
- `python` version 3.9.18 (for visualization)


### Building the benchmarking binary

To build the `build-NAME/knn` binary required to run the benchmarks, run the following commands (replace `CUDA_ARCHITECTURES` with the compute capability of your GPU, e.g., `90` for NVIDIA H100, and `NAME` with the chosen name that distinguishes your build from others; the `NAME` will be used to name the output data as well):

```bash
# Clone the repository and update the submodules:
git clone https://github.com/jiriklepl/bits-knn-jpdc2024.git
cd bits-knn-jpdc2024
git submodule update --init --recursive

# Build the `knn` binary in the build-NAME directory:
./local-build.sh NAME CUDA_ARCHITECTURES build
```

If, for some reason, the `./local-build.sh` script does not work, or you want to apply a specific configuration to the build, you can build the source code manually — consult the [manual-build.md](docs/manual-build.md) file for more information. Note that, even when running multiple build jobs in parallel, the build process can take a long time (up to two hours).


### Collecting the benchmarking data

All benchmarks are run by `run-*.sh` scripts in the `scripts/` directory. These scripts run the `build-NAME/knn` binary with various parameters and output the measured data in the CSV format described in [data-format.md](docs/data-format.md).

The single runs of the benchmarks are conducted by the `build-NAME/knn` binary, which is described in more detail in the [knn.md](docs/knn.md) file that also describes the available configuration options for the proposed algorithms. For the list of all available parameters and the supported algorithms, run the following command:

```bash
build-NAME/knn --help
```

The `local-build.sh` script simplifies redirecting the output to separate files `data/EXPERIMENT-NAME-TIMESTAMP.csv` and `data/EXPERIMENT-NAME-TIMESTAMP.err` (and also to build and test the source code). This way, we can easily distinguish between various builds (via the `NAME` parameter) and runs of the benchmarks (`TIMESTAMP`).


#### (Optional) Tuning the algorithm parameters

To tune the algorithm parameters, run the following commands (replace `CUDA_ARCHITECTURES` with the compute capability of your GPU, e.g., `90` for NVIDIA H100, and `NAME` with the chosen name):

```bash
# To benchmark various values for the items-per-thread parameter of the bits algorithm
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-opt-ipt.sh 128 # 128 is the block size
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-opt-ipt.sh 256 # 256 is the block size
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-opt-ipt.sh 512 # 512 is the block size

# To benchmark various parameters of the variants of the `bits-fused` algorithm
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-fused-params.sh
# fused-cache-params is split into three scripts to reduce the running time of each script:
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-fused-cache-params.sh 1
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-fused-cache-params.sh 2
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-fused-cache-params.sh 4

# To benchmark various parameters of the optimized Bitonic sort algorithms
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-opt-bitonic-sort.sh

# To benchmark various parameters of WarpSelect and BlockSelect algorithms from FAISS
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-params-warp-select.sh
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-params-block-select.sh

# To benchmark various parameters for various distance computation algorithms
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-opt-distances.sh
```

This generates various `data/EXPERIMENT-NAME-TIMESTAMP.csv` and `data/EXPERIMENT-NAME-TIMESTAMP.err` files. Their format is discussed in [data-format.md](docs/data-format.md). Note that these scripts can take a long time (typically up to two hours) to run.

Then, run the following commands to generate `scripts/optima.csv` and `scripts/optima-dist.csv` files, which contain the best-performing parameters for the algorithms (they also automatically prepare a virtual environment `.venv` and install the required Python packages using `pip`):

```bash
# Produce the `scripts/optima.csv` file:
scripts/plot-all.sh analyze-opt

# Produce the `scripts/optima-dist.csv` file:
scripts/plot-all.sh analyze-dist
```

The two files also produce files `scripts/mean-time[-dist]-fixed-max_slowdown.csv` and `scripts/optima[-dist]-fixed-max_slowdown.csv` with the tuning results used to choose one fixed set of parameters for each algorithm (filtered to knn configurations of sufficient query counts) that performs the best across all problem configurations. Each line also shows the slowdown of the run with fixed parameters compared to the best-performing parameters for the given problem configuration (a slowdown of 1.5 would represent a 50% runtime overhead compared to the best-performing run on the given problem configuration). The data in these files are reported in Section 5 in discussions comparing per-configuration tuning to choosing a single fixed parametrization for each algorithm.


#### Running the benchmarks

To run the benchmarks, run the following commands (assuming you have already built and tuned the algorithms; see the previous sections):

```bash
# To run the benchmarks for the optimized Bitonic sort algorithms:
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-bitonic-sort.sh

# To run the benchmarks for comparison of the best-performing Bitonic sort algorithm to the proposed bits algorithm (that adds buffering of filtered items):
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-buffer.sh uniform
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-buffer.sh uniform-ascending
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-buffer.sh uniform-descending

# To run the benchmarks comparing the proposed bits algorithm to State-of-the-Art algorithms for top-k selection:
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-kselection.sh

# To run the benchmarks for distance computation algorithms:
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-distances.sh

# To run the benchmarks for the bits-fused algorithm, comparing it to the bits algorithm and the best-performing distance computation algorithm (MAGMA-distance):
./local-build.sh NAME CUDA_ARCHITECTURES scripts/run-fused.sh
```

Each command runs the given script and redirects its output to the `data/EXPERIMENT-NAME-TIMESTAMP.csv` and `data/EXPERIMENT-NAME-TIMESTAMP.err` files (the format of these files is discussed in [data-format.md](docs/data-format.md)). Each script should run in a reasonable time (within 30 minutes).


### Plotting the results

To plot the results, run the following commands:

```bash
# (Figure 8 in the paper)
# To plot the results for the optimized Bitonic sort algorithms
scripts/plot-all.sh bitonic-sort

# The same script plots the results for the comparison of the best-performing Bitonic sort algorithm to the proposed bits algorithm:

# (Figure 9 in the paper)
# To merge the results from the uniform, uniform-ascending, and uniform-descending experiments:
scripts/plot-all.sh multibuffer

# (Figure 10 in the paper)
# To plot the results for the proposed bits algorithm compared to State-of-the-Art algorithms for top-k selection:
scripts/plot-all.sh kselection > data/kselection-stats.csv

# (Figure 11 in the paper)
# To plot the results for the distance computation algorithms:
scripts/plot-all.sh distances

# (Figure 12 in the paper)
# To plot the results for the bits-fused algorithm (same as the bits algorithm)
scripts/plot-all.sh kselection
```

The plots are stored in the `plots/` directory; the names of the plots correspond to the names of the data files in the `data/` directory (they share the same `EXPERIMENT-NAME-TIMESTAMP`). The only exception to this rule is the plots produced by the `multibuffer` command, which are stored in the `plots/multibuffer-NAME.pdf` files as it merges all results sharing the same `NAME` prefix. All plots are accompanied by a `plots/EXPERIMENT-NAME-TIMESTAMP.csv` file that contains the data being visualized.

The `data/kselection-stats.csv` file contains the statistics for the `kselection` experiment that show the speedup of the proposed algorithm over the state-of-the-art algorithms (maximum, minimum, and average speedup) for each sub-plot and the whole experiment and the relative throughput as a fraction of the throughput limit deduced from the memory bandwidth of the given GPU (the maximum memory throughput divided by the size of one input item in bytes). These reported statistics are used in Section 5.2 of the paper and the Abstract.

Some scripts also produce `plots/extra-EXPERIMENT-NAME-TIMESTAMP.pdf` files that do not filter out algorithms that have not been chosen for the visualization in the paper (as they were superseded by other algorithms).

The figures in the paper show the following plots:

- Figure 8: [`plots/bitonic-sort-hopper01-108431.pdf`](plots/bitonic-sort-hopper01-108431.pdf)
- Figure 9: [`plots/multibuffer-hopper01.pdf`](plots/multibuffer-hopper01.pdf)
- Figure 10: [`plots/kselection-hopper01-108424-Uniform.pdf`](plots/kselection-hopper01-108424-Uniform.pdf)
- Figure 11: [`plots/distances-hopper01-108423.pdf`](plots/distances-hopper01-108423.pdf)
- Figure 12: [`plots/fused-hopper01-108416-Uniform.pdf`](plots/fused-hopper01-108416-Uniform.pdf)

For more information about the setup used to collect the benchmarking results, see the [used-setup.md](docs/used-setup.md) file.


## Demonstration and testing

If you want to quickly demonstrate the proposed algorithms and run the tests validating the correctness of the algorithms, run the following commands (replace `CUDA_ARCHITECTURES` with the compute capability of your GPU, e.g., `90` for NVIDIA H100, and `NAME` with the chosen name that distinguishes your build from others):

```bash
git clone https://github.com/jiriklepl/bits-knn-jpdc2024.git
cd bits-knn-jpdc2024
git submodule update --init --recursive

# Build the `knn-minimal` binary in the build-NAME directory:
./local-build.sh NAME CUDA_ARCHITECTURES minimal-build
```

This builds the `knn-minimal` binary (with reduced configuration options) and the `test` binary that validates the correctness of the proposed algorithms. On a typical system with sufficient memory (32 GiB or more) and at least 16 CPU cores, the build process should take less than 10 minutes. The supported configuration options are discussed in the [knn.md](docs/knn.md) file.

Then, (optionally) run the following commands to run the tests:

```bash
./local-build.sh NAME CUDA_ARCHITECTURES test
```

To demonstrate the efficiency of the proposed algorithms, you can run the following commands:

```bash
# Run the proposed bits algorithm with block size 128 and `items-per-thread` set to 16
build-NAME/knn-minimal -a bits -n 1M -q 1k -k 128 --block-size 128 --items-per-thread 16 --seed 42 -r2 | grep "selection"

# Compare it to Air-topk and GridSelect algorithms (state-of-the-art algorithms)
build-NAME/knn-minimal -a air-topk -n 1M -q 1k -k 128 --seed 42 -r2 | grep "selection"
build-NAME/knn-minimal -a grid-select -n 1M -q 1k -k 128 --seed 42 -r2 | grep "selection"
```

On the standard output, you might see the following lines (the times might vary depending on the hardware):

```csv
bits,uniform,identity,0,1048576,1024,1,128,128,"16,1,1",1,selection,2.8317040000000000e-03
bits,uniform,identity,1,1048576,1024,1,128,128,"16,1,1",1,selection,2.6102669999999999e-03

air-topk,uniform,identity,0,1048576,1024,1,128,128,"1,1,1",1,selection,9.3619640000000004e-03
air-topk,uniform,identity,1,1048576,1024,1,128,128,"1,1,1",1,selection,6.6351589999999998e-03

grid-select,uniform,identity,0,1048576,1024,1,128,128,"1,1,1",1,selection,7.9861005999999998e-02
grid-select,uniform,identity,1,1048576,1024,1,128,128,"1,1,1",1,selection,4.4047710000000000e-03
```

(the outputted Checksums in stderr should be the same for all algorithms; reported times are in seconds - the last column; for a more detailed explanation of the output, see [data-format.md](docs/data-format.md)).

In the example output above, `bits` outperforms the state-of-the-art algorithms `Air-topk` and `GridSelect` on the given problem size (1M points, 1k queries, 128 nearest neighbors to find) with speedup over $2.5\times$ and $1.7\times$, respectively.


## License

The source code and the replication package are licensed under the MIT License. The license text is available in the `LICENSE.txt` file.

Some parts of the source code come from third-party projects. Their licenses are available in the `licenses.txt` file. The relevant parts of the source code are marked with the original license text.
