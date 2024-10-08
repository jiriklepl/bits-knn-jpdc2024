#!/bin/bash -ex

export worker=${1:-ampere01} # NVIDIA Tesla L40 PCIe 40 GB
builder=${builder:-"$worker"}
export CUDA_ARCHITECTURES=${2:-"89"} # L40
export build_dir=build-adalovelace

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
