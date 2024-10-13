#!/bin/bash
#SBATCH -o data/buffer-%N-%j.csv
#SBATCH -e data/buffer-%N-%j.err
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
    n_power=$((PROBLEM_SIZE - q_power))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for k in 32 64 128 256 512 1024 2048; do
        config=$(config_algorithm partial-bitonic-regs $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --seed 24 -a partial-bitonic-regs --block-size "$block_size" -g "$generator" -p "$preprocessor"

        config=$(config_algorithm bits $q $k)
        read -r -a configs <<<"$config"
        block_size=${configs[0]:-256}
        "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" --items-per-thread 1 --seed 24 -a bits --block-size "$block_size" -g "$generator" -p "$preprocessor"
    done
done
