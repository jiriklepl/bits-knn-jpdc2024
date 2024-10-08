#!/bin/bash
#SBATCH -o data/eval-mq-%N-%j.csv
#SBATCH -e data/eval-mq-%N-%j.err
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

seed=24
repeat_count=20

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for config in "1048576 1024" "524288 2048" "262144 4096" "131072 8192"; do
    read -r -a c <<< "$config"
    n=${c[0]}
    q=${c[1]}
    for k in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
        if [ "$k" -ge "$n" ]; then
            break
        fi

        dim=32
        prefix="$knn -r $repeat_count --seed $seed -k $k -n $n -q $q -d $dim"

        if [ "$k" -ge 32 ] && [ "$k" -le 2048 ]; then
            echo "bits" >&2
            $prefix -a bits --items-per-thread $((k < 2048 ? 4 : 8))
        fi
        if [ "$k" -ge 128 ]; then
            if [ "$k" -le 65536 ]; then
                echo "bits-global" >&2
                $prefix -a bits-global --items-per-thread 8
            fi
            echo "two-stage-sample-select" >&2
            $prefix -a two-stage-sample-select
        fi
    done

    # distances and the fused kernel
    for dim in 2 4 8 16 32 64 128; do
        "$knn" -r "$repeat_count" --seed "$seed" -n "$n" -q "$q" -d "$dim" -a magma-dist

        # fused kernel
        for k in 4 8 16 32 64; do
            if [ "$dim" -lt 128 ]; then
                "$knn" -r "$repeat_count" --seed "$seed" -k "$k" -n "$n" -q "$q" -d "$dim" -a fused-regs
            fi
        done
    done
done
