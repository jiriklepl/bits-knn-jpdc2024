#!/bin/bash

# This script is to be sourced from *-build.sh scripts in the root directory of the project.

set -e

if [ -z "$build_dir" ]; then
    echo "Error: build_dir is not set."
    exit 1
fi

if [ "$3" == "build" ] || [ "$3" == "minimal-build" ] || [ "$3" == "all" ] || [ -z "$3" ]; then
    run_single "$builder" cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -S .

    if [ "$3" == "minimal-build" ]; then
        run_single "$builder" cmake --build "$build_dir" --config Release --parallel 16 -t knn-minimal
        run_single "$builder" cmake --build "$build_dir" --config Release --parallel 16 -t test
        exit 0
    fi

    run_single "$builder" cmake --build "$build_dir" --config Release --parallel 16 -t knn
    run_single "$builder" cmake --build "$build_dir" --config Release --parallel 16 -t test

    run_single "$worker" "$build_dir"/test

    if [ "$3" == "build" ]; then
        exit 0
    fi
elif [ "$3" == "test" ]; then
    run_single "$worker" "$build_dir"/test
    exit 0
elif [ -n "$3" ]; then
    shift 2
    run_batch "$@"
    exit 0
fi

run_batch scripts/run-distances.sh
run_batch scripts/run-bits-sq.sh
run_batch scripts/run-eval-mq.sh
run_batch scripts/run-fused.sh
run_batch scripts/run-kselection-sp.sh
run_batch scripts/run-opt-bitonic-sort.sh
run_batch scripts/run-opt-buffer.sh
run_batch scripts/run-opt-ipt.sh
run_batch scripts/run-opt-two-stage.sh
run_batch scripts/run-parallel.sh
run_batch scripts/run-params-warp-select.sh

# May contain impossible configurations, for tuning:
run_batch scripts/run-fused-cache-params.sh
run_batch scripts/run-fused-params.sh
run_batch scripts/run-params-block-select.sh
