#!/bin/bash
# Sourced by the token-sampling and gradient-compression batch entry points.
set -eo pipefail
application=${application:?Source this helper through an application batch entry point}
root_dir=${root_dir:?Source this helper through an application batch entry point}

default_manifest="$root_dir/data/application-inputs/$application/manifest.json"
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
Capture the model input before submitting the GPU job, or pass an existing manifest.
See docs/model-applications.md for the pinned exporter setup and:
  .venv-model/bin/python scripts/export-application-inputs.py $application --output data/application-inputs/$application"
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

# Publish the CSV only after every k, block size and batch size has passed verification.
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
for k in "$@"; do
    for block_size in 128 256 512; do
        for items_per_thread in 4 7 8 13 16; do
            backends=bits-prefetch,bits-sq
            # The comparison backends ignore both BITS tuning options; run them once per k.
            if [ "$block_size" -eq 512 ] && [ "$items_per_thread" -eq 16 ]; then
                backends+=,air-topk,grid-select,block-select
            fi
            python3 "$root_dir/scripts/run-applications.py" "$manifest" \
                --binary "$build_dir/$application" --backends "$backends" \
                --degree 32 --bits-block-size "$block_size" \
                --items-per-thread "$items_per_thread" --k "$k" --warmup 10 --repeat 30 \
                > "$temporary/current.csv"
            if [ -s "$temporary/timings.csv" ]; then
                sed '1d' "$temporary/current.csv" >> "$temporary/timings.csv"
            else
                cat "$temporary/current.csv" > "$temporary/timings.csv"
            fi
        done
    done
done
cat "$temporary/timings.csv"
