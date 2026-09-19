#!/bin/bash
#SBATCH -o data/database-topn-%N-%j.csv
#SBATCH -e data/database-topn-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --exclusive

set -eo pipefail

root_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}"
default_manifest="$root_dir/data/application-inputs/tpch-sf01/manifest.json"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    printf 'Usage: %s [MANIFEST [K ...]]\nDefault manifest: %s\nDefault K: 32 64 128 256 512 1024\n' \
        "$0" "$default_manifest"
    exit 0
fi
# shellcheck source=scripts/config.sh
. "$root_dir/scripts/config.sh"

manifest="${1:-$default_manifest}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ ! -f "$manifest" ]; then
    print_error "Input manifest not found: $manifest
Pass an existing manifest as the first argument, or prepare the default input from the repository root:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r scripts/requirements-applications.txt
  .venv/bin/python scripts/export-application-inputs.py database-topn --scale-factor 0.1 --output data/application-inputs/tpch-sf01"
fi
if [ "$#" -eq 0 ]; then
    set -- 32 64 128 256 512 1024
fi
declare -A seen
for k in "$@"; do
    case "$k" in
        32|64|128|256|512|1024) ;;
        *) print_error "All five backends require K = 32, 64, 128, 256, 512 or 1024" ;;
    esac
    if [ -n "${seen[$k]}" ]; then
        print_error "Duplicate K: $k"
    fi
    seen[$k]=1
done

# Publish the CSV only after every k has completed and passed verification.
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
for k in "$@"; do
    python3 "$root_dir/scripts/run-applications.py" "$manifest" \
        --binary "$build_dir/database-topn" \
        --backends bits,bits-sq,air-topk,grid-select,block-select \
        --degree 32 --bits-block-size 512 --k "$k" --warmup 10 --repeat 30 \
        > "$temporary/current.csv"
    if [ -s "$temporary/timings.csv" ]; then
        sed '1d' "$temporary/current.csv" >> "$temporary/timings.csv"
    else
        cat "$temporary/current.csv" > "$temporary/timings.csv"
    fi
done
cat "$temporary/timings.csv"
