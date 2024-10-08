#!/bin/bash
#SBATCH -o data/params-warp-select-%N-%j.csv
#SBATCH -e data/params-warp-select-%N-%j.err
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
block_size=128

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for n in 128 512 1024; do
    q=$((2048 / n))

    for k in 32 64 128 256 512 1024 2048; do
        for thread_queue in 2 3 4 5 6 7 8 9 10; do
            "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 --block-size "$block_size" --items-per-thread "$thread_queue" -a warp-select-tunable
        done

        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 -a warp-select
    done
done
