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
PROBLEM_SIZE=25

# we try to keep the ammount of work constant unless the input would be too large
total_work=$((64 * 1024 * 32 * 1024 * 128))
max_gb=16 # max memory in GiB
max_floats=$((max_gb * 1024 * 1024 * 1024 / 4))

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 10 11 12 13; do
    # choose n_power so that sqrt(q) * N == 2^PROBLEM_SIZE
    n_power=$((PROBLEM_SIZE - q_power / 2))

    q=$((2 ** q_power))
    n=$((2 ** n_power))

    for dim in 4 8 16 32 64 128; do
        for block_size in 64 128 256 512; do
            # baseline-dist always uses a column-major layout for input matrices
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 -a baseline-dist
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 --point-layout column -a cublas-dist
        done

        # baseline-dist always uses a column-major layout for input matrices
        for block_size in 8 16 32 64; do
            "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --block-size "$block_size" --seed 24 -a tiled-dist
        done

        # magma-dist always uses a column-major layout for input matrices
        "$knn" -r "$repeat_count" -d "$dim" -n "$n" -q "$q" --seed 24 -a magma-dist
    done
done
