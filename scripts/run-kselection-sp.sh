#!/bin/bash
#SBATCH -o data/kselection-sp-%N-%j.csv
#SBATCH -e data/kselection-sp-%N-%j.err
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

gpu_deduce

"$knn" --header
for n in 32 64 128 256 512 1024; do
    q=$((1024 / n))

    for k in 32 64 128 256 512 1024 2048; do
        config=$(config_algorithm bits $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 --items-per-thread "$items_per_thread" -a bits --block-size "$block_size" --deg "$deg"

        config=$(config_algorithm bits-sq $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 --items-per-thread "$items_per_thread" -a bits-sq --block-size "$block_size" --deg "$deg"

        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 -a warp-select-tuned
        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 -a warp-select
        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 -a block-select
        "$knn" -r "$repeat_count" -n "$n"k -q "$q"k -k "$k" --seed 24 -a block-select-tuned
    done
done
