#!/bin/bash
#SBATCH -o data/opt-buffer-%N-%j.csv
#SBATCH -e data/opt-buffer-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=20
PROBLEM_SIZE=25

gpu_deduce

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 6 8 10 12; do
    # choose n_power so that sqrt(q) * N == 2^PROBLEM_SIZE
    n_power=$((PROBLEM_SIZE - q_power / 2))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for k in 32 64 128 256 512 1024 2048; do
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --seed 24 -a partial-bitonic-regs
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread 1 --seed 24 -a bits
    done
done
