#!/bin/bash
#SBATCH -o data/distances-%N-%j.csv
#SBATCH -e data/distances-%N-%j.err
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
for q_power in 10 11 12 13; do
    n_power=$((PROBLEM_SIZE - q_power))

    q=$((2 ** q_power))
    n=$((2 ** n_power))

    for dim in 4 8 16 32 64 128 256; do
        config=$(config_distance baseline-dist $q $dim)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        # baseline-dist always uses a column-major layout for input matrices
        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 -a baseline-dist

        config=$(config_distance tiled-dist $q $dim)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-16}
        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 -a tiled-dist

        config=$(config_distance cublas-dist $q $dim)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 --point-layout column -a cublas-dist

        # magma-dist always uses a column-major layout for input matrices
        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --seed 24 -a magma-dist

        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --seed 24 -a magma-part-dist
    done
done
