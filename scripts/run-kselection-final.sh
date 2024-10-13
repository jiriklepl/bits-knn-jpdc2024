#!/bin/bash
#SBATCH -o data/kselection-query-%N-%j.csv
#SBATCH -e data/kselection-query-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-long
#SBATCH --time=6:00:00
#SBATCH --mem=0
#SBATCH --exclusive

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

variant=${1:-uniform}

if [ -z "$variant" ]; then
    trap 'echo "Usage: $0 [<variant>]"' EXIT
    print_error "No variant specified"
fi

generator=uniform
preprocessor=identity

case "$variant" in
uniform-ascending*)
    generator=uniform
    preprocessor=ascending
    ;;
uniform-descending*)
    generator=uniform
    preprocessor=descending
    ;;
uniform:* | uniform)
    generator=${variant}
    preprocessor=identity
    ;;
normal:* | normal)
    generator=${variant}
    preprocessor=identity
    ;;
radix-adversarial:* | radix-adversarial)
    generator=${variant}
    preprocessor=identity
    ;;
*)
    print_error "Unknown variant: $variant"
    ;;
esac

repeat_count=20
PROBLEM_SIZE=30

gpu_deduce

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 6 8 10 12; do
    q=$((2 ** q_power))

    n_power=$((PROBLEM_SIZE - q_power))

    for k_power in {5..11}; do
        N=$((2 ** n_power))
        k=$((2 ** k_power))

        # not enough GPU memory on V100
        if [ $((q * N)) -gt $((2 ** 32)) ]; then
            continue
        fi

        if [ $k -gt $N ]; then
            continue
        fi

        config=$(config_algorithm bits $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 --items-per-thread "$items_per_thread" -a bits -g "$generator" -p "$preprocessor" --block-size "$block_size" --deg "$deg"

        config=$(config_algorithm bits-prefetch $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        items_per_thread=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 --items-per-thread "$items_per_thread" -a bits-prefetch -g "$generator" -p "$preprocessor" --block-size "$block_size" --deg "$deg"

        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 -a air-topk -g "$generator" -p "$preprocessor"
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 -a grid-select -g "$generator" -p "$preprocessor"
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 -a radik -g "$generator" -p "$preprocessor"

        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 -a warp-select -g "$generator" -p "$preprocessor"
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 -a block-select -g "$generator" -p "$preprocessor"

        config=$(config_algorithm block-select-tunable $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        thread_queue=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 --items-per-thread "$thread_queue" -a block-select-tunable -g "$generator" -p "$preprocessor" --block-size "$block_size" --deg "$deg"

        config=$(config_algorithm warp-select-tunable $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        deg=${configs[1]:-1}
        thread_queue=${configs[2]:-1}
        "$knn" -r "$repeat_count" -n "$N" -q "$q" -k "$k" --seed 24 --items-per-thread "$thread_queue" -a warp-select-tunable -g "$generator" -p "$preprocessor" --block-size "$block_size" --deg "$deg"
    done
done
