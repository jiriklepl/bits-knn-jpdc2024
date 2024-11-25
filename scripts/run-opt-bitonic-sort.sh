#!/bin/bash
#SBATCH -o data/opt-bitonic-sort-%N-%j.csv
#SBATCH -e data/opt-bitonic-sort-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --exclusive

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=20
PROBLEM_SIZE=30

gpu_deduce

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 6 8 10 12; do
    n_power=$((PROBLEM_SIZE - q_power))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for k in 32 64 128 256 512 1024 2048; do
        for block_size in 64 128 256 512; do
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 17 -a partial-bitonic
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 17 -a partial-bitonic-warp
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 17 -a partial-bitonic-warp-static
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 17 -a partial-bitonic-regs
        done
    done
done
