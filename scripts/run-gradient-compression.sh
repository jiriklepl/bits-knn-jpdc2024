#!/bin/bash
#SBATCH -o data/gradient-compression-%N-%j.csv
#SBATCH -e data/gradient-compression-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --exclusive

root_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}"
application=gradient-compression
# shellcheck source=scripts/tensor-benchmark.sh
. "$root_dir/scripts/tensor-benchmark.sh"
