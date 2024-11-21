# KNN binary

This file describes the `knn` binary that is used to run the experiments evaluating the top-k selection algorithms and the distance computation algorithms as described in [README.md](../README.md).

The build process for the `knn` binary is described in [README.md](../README.md) and [manual-build.md](manual-build.md).

## Usage

The `knn` binary (assuming it was built in the `build-NAME` directory) is run with the following command:

```bash
./build-NAME/knn [OPTIONS]
```

where `OPTIONS` are the following:

```bash
  -a, --algorithm arg         Used algorithm (default: serial)
  -r, --repeat arg            Number of executions (default: 1)
  -k, --knn arg               Number of nearest neighbors (default: 1)
  -n, --number arg            Number of objects in the database (default:
                              1024)
  -q, --query arg             Number of query objects (default: 1024)
  -d, --dimension arg         Dimension of objects (default: 1)
  -g, --generator arg         Data generator (default: uniform)
  -p, --preprocessor arg      Data preprocessor (default: identity)
      --seed arg              Seed for the data generator (default: 0)
      --block-size arg        Block size for CUDA kernels (default: 256)
      --items-per-thread arg  Number of items per thread in the fused-regs
                              kernel (two comma separated values) (default:
                              1)
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
      --header                Print header with column names and exit
  -h, --help                  Show help message and exit
```

The above options and the comprehensive list of available algorithms can be displayed running the binary with the `--help` option:

```bash
./build-NAME/knn --help
```

The names of the evaluated algorithms are described in [algorithms.md](algorithms.md) that maps the algorithm names as used in the paper and visualizations to the `-a ALGORITHM` argument values and their corresponding implementations in the source code.


## Changing the available configurations

The available configurations for the proposed algorithms are defined in the `src/gen.py` script (see [gen.py](../src/gen.py)) that generates files that define the configurations for the `bits` and `bits-fused` algorithms (and tuned versions of `WarpSelect` and `BlockSelect` algorithms from the FAISS library). Note that after changing the configurations and generating the files, the `knn` binary must be rebuilt to include the new configurations; which may take a significant amount of time.


## Example

The single runs of the benchmarks are conducted by the `./build-NAME/knn` binary as follows:

```bash
./build-NAME/knn -a bits-prefetch --generator uniform --preprocessor identity -n 10M -q 1k -k 128 -d 1 -r 20 --seed 42 --block-size 256 --items-per-thread 32
```

This command runs the `bits` algorithm with prefetching enabled on the uniform data generator with no preprocessing, 10 MiB of database points, 1 KiB of queries, 128 nearest neighbors to find, 1-dimensional points, 20 repetitions, random seed 42 (for reproducibility), block size 256, and 32 items per thread (in each batch of database items).

A sequence of the above commands is usually preceded by the following command that prints the header with the column names:

```bash
./build-NAME/knn --header
```

## Output

The `knn` binary outputs the runtime of each phase of the algorithm in seconds for each repetition. The output is in the standard output in the CSV format described in [data-format.md](data-format.md) accompanied by the checksums and error messages in the standard error output.
