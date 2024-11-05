#!/bin/bash
#SBATCH -o data/ipt-%N-%j.csv
#SBATCH -e data/ipt-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-long
#SBATCH --time=4:00:00

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
        config=$(config_algorithm bits $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread "$items_per_thread" --deg "$deg" --seed 23 -a bits --block-size "$block_size"

        config=$(config_algorithm bits-prefetch $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread "$items_per_thread" --deg "$deg" --seed 23 -a bits-prefetch --block-size "$block_size"
    done
done
