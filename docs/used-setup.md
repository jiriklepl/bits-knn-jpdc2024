# Setup used to collect the benchmarking results

In our `gpulab` cluster ([https://gitlab.mff.cuni.cz/mff/hpc/clusters](https://gitlab.mff.cuni.cz/mff/hpc/clusters)), we use the [Slurm Workload Manager](https://slurm.schedmd.com/documentation.html) to schedule the benchmarks on cluster nodes equipped with NVIDIA GPUs. For the Slurm setup, we use the following scripts:

- `volta-build.sh` for the NVIDIA Tesla V100 (Volta) GPU-equipped node; sets `CUDA_ARCHITECTURES=70` and produces the `*-volta05-*.csv` results
- `ampere-build.sh` for the NVIDIA A100 (Ampere) GPU-equipped node; sets `CUDA_ARCHITECTURES=80` and produces the `*-ampere02-*.csv` results
- `hopper-build.sh` for the NVIDIA H100 (Hopper) GPU-equipped node; sets `CUDA_ARCHITECTURES=89` and produces the `*-hopper01-*.csv` results
- `lovelace-build.sh` for the NVIDIA L40 (Lovelace) GPU-equipped node; sets `CUDA_ARCHITECTURES=90` and produces the `*-ampere01-*.csv` results

All these scripts follow the same pattern as the `local-build.sh` script in the [Results reproduction](../README.md#results-reproduction) section of the README.md file. The core functionality shared by all these scripts (including the `local-build.sh` script) is in the `scripts/executor.sh` script. The difference between the `local-build.sh` and the other scripts is in that the former runs the benchmarks directly on the node where it is executed, while the others submit the benchmarks to the Slurm queueing system via the `sbatch` command. For the slurm scripts, the `TIMESTAMP` is replaced by the job ID assigned by the Slurm queueing system.

The scripts are run as follows:

```bash
# giving example for the hopper-build.sh script that uses the `./build-hopper/knn` binary
./hopper-build.sh hopper 90 ./scripts/run-kselection.sh
```

The above command runs the `./scripts/run-kselection.sh` script on the `hopper01` node with the `CUDA_ARCHITECTURES=90` setting and produces the `data/kselection-hopper01-JOBID.csv` and `data/kselection-hopper01-JOBID.err` files as described in the [data-format.md](data-format.md) file.
