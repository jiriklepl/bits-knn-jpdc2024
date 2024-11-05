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

repeat_count=30
n=512k

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q in 64 256 512 1024 2048 4096 8192; do
    for block_size in 64 128 256; do
        for k in 32 64 128 256 512 1024 2048; do
            for thread_queue in 2 3 4 5 6 7 8 9 10; do
                "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --seed 13 --block-size "$block_size" --items-per-thread "$thread_queue" -a warp-select-tunable
            done
        done
    done
done
