#!/bin/bash -ex

export worker=${1:-volta05} # NVIDIA Tesla V100 32 GB SXM2
builder=${builder:-"$worker"}
export CUDA_ARCHITECTURES=${2:-"70"} # V100
export build_dir=build-volta

export account=kdss
partition=gpu-short
long_time=2:00:00

run_batch() {
    sbatch -w "$worker" --export=worker,CUDA_ARCHITECTURES,build_dir,account "$@"
}

run_single() {
    srun -A "$account" -p "$partition" --gres=gpu:1 -w "$builder" -t "$long_time" -c 16 -n 1 -- \
        "$@"
}

. ./scripts/executor.sh
