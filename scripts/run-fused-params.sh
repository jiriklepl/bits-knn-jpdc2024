#!/bin/bash
#SBATCH -o data/fused-params-%N-%j.csv
#SBATCH -e data/fused-params-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-long
#SBATCH --time=4:00:00

root_dir="$SLURM_SUBMIT_DIR"
if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")/..")
fi

. "$root_dir/scripts/config.sh"

repeat_count=30
n=256k

if [ -z "$knn" ]; then
    print_error "knn executable not set"
fi

"$knn" --header
for q_power in 8 9 10 11 12 13; do
    q=$((2 ** q_power))

    for dim in 4 8 16 32 64; do
        for k in 2 4 8 16 32 64 128; do
            for bs in 128 256 512; do
                "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --block-size "$bs" --seed 24 -a fused-tc-half

                if [ "$CUDA_ARCHITECTURES" -ge 80 ]; then
                    "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --block-size "$bs" --seed 24 -a fused-tc-bfloat16
                    "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --block-size "$bs" --seed 24 -a fused-tc-double
                fi
            done

            # for rq in 2 4 8; do
            #     for bs in 4 8 16; do
            #         "$knn" -r "$repeat_count" -n "$n" -q "$q" -k "$k" -d "$dim" --items-per-thread "$rq,4" --block-size "$bs" --seed 24 -a fused-regs-tunable
            #     done
            # done
        done
    done
done
