#!/bin/bash
#SBATCH -o data/token-sampling-%N-%j.csv
#SBATCH -e data/token-sampling-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --exclusive

root_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}"
application=token-sampling
# shellcheck source=scripts/tensor-benchmark.sh
. "$root_dir/scripts/tensor-benchmark.sh"
