# Setup used to collect the benchmarking results

The [Chimera laboratory](https://gitlab.mff.cuni.cz/mff/hpc/clusters) uses Slurm to schedule benchmarks on GPU nodes. Its build wrappers are in [scripts/chimera](../scripts/chimera):

| Script | Default node | GPU | Build directory |
| --- | --- | --- | --- |
| `volta-build.sh` | `volta05` | NVIDIA Tesla V100 | `build-volta` |
| `ampere-build.sh` | `ampere02` | NVIDIA A100 | `build-ampere` |
| `adalovelace-build.sh` | `ampere01` | NVIDIA L40 | `build-adalovelace` |
| `hopper-build.sh` | `hopper01` | NVIDIA H100 | `build-hopper` |
| `bw-build.sh` | `bw01` | NVIDIA RTX PRO 6000 Blackwell Server Edition | `build-bw` |

Run these scripts from the repository root. They accept the same `NODE ARCH ACTION` arguments as [local-build.sh](../local-build.sh), with `native` as the default CUDA architecture. Builds and tests run through `srun`; benchmark scripts are submitted through `sbatch`.

```bash
./scripts/chimera/hopper-build.sh hopper01 native build
./scripts/chimera/hopper-build.sh hopper01 native scripts/run-kselection.sh
```

This builds into `build-hopper` and writes the benchmark results to `data/kselection-hopper01-JOBID.csv`, with diagnostics in the corresponding `.err` file. Slurm job IDs replace the timestamps used by local runs; see [data-format.md](data-format.md) for the CSV format.

For the three application benchmarks, use the `applications-build` and `applications-run` actions described in [model-applications-commands.md](model-applications-commands.md).
