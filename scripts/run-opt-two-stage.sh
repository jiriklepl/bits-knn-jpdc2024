#!/bin/bash
#SBATCH -o data/opt-two-stage-%N-%j.csv
#SBATCH -e data/opt-two-stage-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=20
n=1024
q=1

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header

# determine baseline
"$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k 512k --seed 24 -a cub-sort
