#!/bin/bash -ex

export worker=${1:-bw01} # NVIDIA RTX PRO 6000 Blackwell Server Edition
builder=${builder:-"$worker"}
export CUDA_ARCHITECTURES=${2:-"native"}
export build_dir=build-volta

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

    srun -A "$account" -p "$partition" --gres=gpu:1 -w "$runner" -t "$long_time" -c 16 -n 1 -- "$@"
}

. ./scripts/executor.sh
