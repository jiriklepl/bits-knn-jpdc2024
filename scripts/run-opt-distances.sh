#!/bin/bash
#SBATCH -o data/opt-distances-%N-%j.csv
#SBATCH -e data/opt-distances-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-long
#SBATCH --time=6:00:00

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=20
PROBLEM_SIZE=30

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 10 11 12 13; do
    n_power=$((PROBLEM_SIZE - q_power))

    q=$((2 ** q_power))
    n=$((2 ** n_power))

    for dim in 4 8 16 32 64 128 256; do
        for block_size in 64 128 256 512; do
            # baseline-dist always uses a column-major layout for input matrices
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 13 -a baseline-dist
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 13 --point-layout column -a cublas-dist
        done

        # baseline-dist always uses a column-major layout for input matrices
        for block_size in 8 16 32 64; do
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 13 -a tiled-dist
        done
    done
done
