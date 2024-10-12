#!/bin/bash
#SBATCH -o data/bitonic-sort-%N-%j.csv
#SBATCH -e data/bitonic-sort-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00

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
for q_power in 6 8 10 12; do
    n_power=$((PROBLEM_SIZE - q_power))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for k in 32 64 128 256 512 1024 2048; do
        config=$(config_algorithm partial-bitonic $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 24 -a partial-bitonic

        # partial-bitonic-warp is a suboptimal variant of partial-bitonic-warp-static
        # config=$(config_algorithm partial-bitonic-warp $q $k)
        # read -r -a configs <<<"$config"
        # block_size=${configs[0]:-256}
        # "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 24 -a partial-bitonic-warp

        config=$(config_algorithm partial-bitonic-warp-static $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 24 -a partial-bitonic-warp-static

        config=$(config_algorithm partial-bitonic-regs $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --block-size "$block_size" --seed 24 -a partial-bitonic-regs
    done
done
