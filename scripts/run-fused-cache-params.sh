#!/bin/bash
#SBATCH -o data/fused-cache-params-%N-%j.csv
#SBATCH -e data/fused-cache-params-%N-%j.err
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

deg=${1:-256}

case "$deg" in
1 | 2 | 4)
    ;;
*)
    print_error "Invalid deg (first cmd arg): $deg"
    ;;
esac

repeat_count=20
PROBLEM_SIZE=30

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 8 9 10 11 12 13; do
    n_power=$((PROBLEM_SIZE - q_power))

    n=$((2 ** n_power))
    q=$((2 ** q_power))

    for dim in 4 8 16; do
        for k in 8 16 32 64 128 256; do
            # for rq in 2 4 8 16; do
            for rq in 4 8; do # prune 2, 16
                # for rn in 4 8 16; do
                for rn in 4 8; do # prune 16
                    for rd in 1 2 4; do
                        # for bq in 1 2 4; do
                        for bq in 1 2; do # prune 4
                            "$knn" --point-layout column --query-layout column -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --items-per-thread "$rq,$rn,$rd" --deg "$deg" --block-size "$bq" --seed 11 -a fused-cache
                        done
                    done
                done
            done
        done
    done
done
