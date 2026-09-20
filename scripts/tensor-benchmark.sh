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

temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
for k in "$@"; do
    python3 "$root_dir/scripts/run-applications.py" "$manifest" \
        --binary "$build_dir/$application" \
        --backends bits-prefetch,bits-sq,air-topk,grid-select,block-select \
        --degree 32 --bits-block-size 512 --k "$k" --warmup 10 --repeat 30 \
        > "$temporary/current.csv"
    if [ -s "$temporary/timings.csv" ]; then
        sed '1d' "$temporary/current.csv" >> "$temporary/timings.csv"
    else
        cat "$temporary/current.csv" > "$temporary/timings.csv"
    fi
done
cat "$temporary/timings.csv"
