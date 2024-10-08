#!/bin/bash
#SBATCH -o data/parallel-%N-%j.csv
#SBATCH -e data/parallel-%N-%j.err
#SBATCH -c 16
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

n="2m"
q="1k"

gpu_deduce

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for dim in 4 8 16 32 64; do
    for k in 32 64 128 256 512 1024 2048; do
        config=$(config_algorithm bits $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r 20 -d "$dim" -n "$n" -q "$q" -k "$k" --seed 24 -a bits --items-per-thread "$items_per_thread" --block-size "$block_size" --deg "$deg"

        config=$(config_algorithm bits-sq $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r 20 -d "$dim" -n "$n" -q "$q" -k "$k" --seed 24 -a bits-sq --items-per-thread "$items_per_thread" --block-size "$block_size" --deg "$deg"

        "$knn" -r 20 -d "$dim" -n "$n" -q "$q" -k "$k" --seed 24 -a parallel
    done
done
