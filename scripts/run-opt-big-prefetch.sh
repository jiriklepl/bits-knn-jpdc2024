#!/bin/bash
#SBATCH -o data/opt-prefetch-%N-%j.csv
#SBATCH -e data/opt-prefetch-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=20
n=2048
q=1000

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for k in 32 64 128 256 512 1024 2048; do
    for batch_size in $(seq 16); do
        "$knn" -r "$repeat_count" -n "$n"k -q "$q" -k "$k" --items-per-thread "$batch_size" --seed 17 -a bits
        "$knn" -r "$repeat_count" -n "$n"k -q "$q" -k "$k" --items-per-thread "$batch_size" --seed 17 -a bits-prefetch
    done
done
