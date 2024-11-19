#!/bin/bash -ex

export worker=${1:-ampere02} # NVIDIA Tesla A100 PCIe 80 GB
builder=${builder:-"$worker"}
export CUDA_ARCHITECTURES=${2:-"80"} # A100
export build_dir=build-ampere

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
