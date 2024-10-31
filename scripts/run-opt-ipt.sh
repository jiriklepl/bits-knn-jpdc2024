#!/bin/bash
#SBATCH -o data/opt-ipt-%N-%j.csv
#SBATCH -e data/opt-ipt-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-long
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --exclusive

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

block_size=${1:-256}

case "$block_size" in
128 | 256 | 512)
    ;;
*)
    print_error "Invalid block size: $block_size"
    ;;
esac

repeat_count=30
n=512k

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q in 64 256 512 1024 2048 4096 8192; do
    for k in 2 4 8 16 32 64 128 256 512 1024 2048; do
        for batch_size in $(seq 16); do
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread "$batch_size" --deg 1 --seed 17 -a bits --block-size "$block_size"
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread "$batch_size" --deg 1 --seed 17 -a bits-prefetch --block-size "$block_size"
        done
    done
done
