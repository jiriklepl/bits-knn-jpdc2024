#!/bin/bash -ex

export worker=${1:-hopper01}         # NVIDIA H100 PCIe 40 GB
builder=${builder:-"volta05"}        # NVIDIA Tesla V100 PCIe 32 GB
export CUDA_ARCHITECTURES=${2:-"90"} # H100
export build_dir=build-hopper

export account=kdss
partition=gpu-short
long_time=2:00:00

run_batch() {
    sbatch -w "$worker" --export=worker,CUDA_ARCHITECTURES,build_dir,account "$@"
}

# $1 == runner
run_single() {
    runner=${1:-"$worker"}
    shift

    srun -A "$account" -p "$partition" --gres=gpu:1 -w "$runner" -t "$long_time" -c 16 -n 1 -- \
        "$@"
}

. ./scripts/executor.sh
