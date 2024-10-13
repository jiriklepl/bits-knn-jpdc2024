#!/bin/bash
#SBATCH -o data/fused-%N-%j.csv
#SBATCH -e data/fused-%N-%j.err
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

repeat_count=20
PROBLEM_SIZE=30

gpu_deduce

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 10 11 12 13; do
    n_power=$((PROBLEM_SIZE - q_power))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for dim in 4 8 16; do
        for k in 2 4 8 16 32 64 128; do
            config=$(config_algorithm bits-prefetch $q $k)
            read -r -a configs <<<"$config"
            block_size=${configs[0]:-256}
            deg=${configs[1]:-1}
            items_per_thread=${configs[2]:-1}
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --seed 24 -a bits-prefetch --items-per-thread "$items_per_thread" --block-size "$block_size" --deg "$deg"

            if [ "$k" -le 64 ]; then
                "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --seed 24 -a rapidsai-fused
            fi

            # config=$(config_algorithm fused-regs-tunable $q $k)
            # read -r -a configs <<<"$config"
            # block_size=${configs[0]:-256}
            # items_per_thread=${configs[2]:-1}
            # items_per_thread2=${configs[3]:-1}
            # "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --seed 24 -a fused-regs-tunable --items-per-thread "$items_per_thread,$items_per_thread2" --block-size "$block_size"

            # config=$(config_algorithm fused-tc-half $q $k $dim)
            # read -r -a configs <<<"$config"
            # block_size=${configs[0]:-256}
            # "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --seed 24 -a fused-tc-half --block-size "$block_size"

            config=$(config_algorithm fused-cache $q $k $dim)
            read -r -a configs <<<"$config"
            block_size=${configs[0]:-1}
            deg=${configs[1]:-1}
            items_per_thread=${configs[2]:-1}
            items_per_thread2=${configs[3]:-1}
            items_per_thread3=${configs[4]:-1}
            "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --seed 24 -a fused-cache --items-per-thread "$items_per_thread,$items_per_thread2,$items_per_thread3" --block-size "$block_size" --deg "$deg"
        done
    done
done
