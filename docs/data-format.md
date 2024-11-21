# Data format

Each command starting with `./local-build.sh` discussed in [Results reproduction](../README.md#results-reproduction) runs the given `./scripts/run-EXPERIMENT.sh` script (e.g., `./scripts/run-kselection.sh` for the top-k selection experiments) and redirects its output to the `./data/EXPERIMENT-NAME-TIMESTAMP.csv` and `./data/EXPERIMENT-NAME-TIMESTAMP.err` files.

The files are named `EXPERIMENT-NAME-TIMESTAMP.csv` and `EXPERIMENT-NAME-TIMESTAMP.err`, where:

- `EXPERIMENT` is the name of the experiment (e.g., `kselection`)
- `NAME` is the name of the build identifying the given system and its configuration (in the collected data, one such build is `volta05` that identifies a node with an NVIDIA Tesla V100 GPU on our cluster)
  - given the name `NAME`, the `./local-build.sh` script either builds or uses the `./build-NAME/knn` binary
- `TIMESTAMP` is the timestamp of the experiment to distinguish between multiple runs of the same experiment; in our cluster, we use Slurm to schedule the experiments, and the timestamp is represented by the Slurm job ID

The `.err` files contain checksums (sums of the indices of the database points for each query, `mod 2^32`) for the generated data and report incorrect parameters and parameters that are not supported by the given algorithm/hardware configuration. Note that since some of the distances may be equal, the checksums are not guaranteed to be fully reliable for rigorous verification; however, these situations are quite rare and even in such cases, the checksums give a close result (usually differing by few points). For verification, the `-v GOLDEN_IMPLEMENTATION` option can be used to compare the results of one implementation to a "golden" implementation (e.g., the serial implementation).

All the `.csv` files report runtimes in seconds for each phase of the computation and the parameters of the experiments.They follow a uniform format:

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
