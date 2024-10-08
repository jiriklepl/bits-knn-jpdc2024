#!/bin/bash
#SBATCH -o data/bits-sq-%N-%j.csv
#SBATCH -e data/bits-sq-%N-%j.err
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
n="128k"

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header | sed "s/\$/,deg/"
for deg in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
    for k in 32 64 128 256 512 1024 2048; do
        "$knn" -r "$repeat_count" -d 16 -n "$n" -q 1 -k "$k" --seed 24 -a bits-sq --deg "$deg" | sed "s/\$/,$deg/"
    done
done
