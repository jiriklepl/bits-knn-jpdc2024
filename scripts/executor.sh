#!/bin/bash

# This script is to be sourced from *-build.sh scripts in the root directory of the project.

set -e

NPROC=${NPROC:-"$(nproc)"}
worker=${worker:-"$(hostname)"}
builder=${builder:-"$(hostname)"}

if [ -z "$build_dir" ]; then
    echo "Error: build_dir is not set."
    exit 1
fi

test_database() {
    run_single "$worker" "$build_dir"/test-applications '[applications]'
    run_single "$worker" env DATABASE_TOPN_BINARY="$build_dir/database-topn" \
        PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/applications -p 'test_*.py'
}

if [ "$3" == "build" ] || [ "$3" == "minimal-build" ] || [ "$3" == "database-build" ] || [ "$3" == "all" ] || [ -z "$3" ]; then
    run_single "$builder" cmake  -S . -B "$build_dir" -D CMAKE_BUILD_TYPE=Release -D CMAKE_EXPORT_COMPILE_COMMANDS=ON -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"

    if [ "$3" == "minimal-build" ]; then
        run_single "$builder" cmake --build "$build_dir" --config Release --parallel "$NPROC" -t knn-minimal
        run_single "$builder" cmake --build "$build_dir" --config Release --parallel "$NPROC" -t test
        exit 0
    fi

    run_single "$builder" cmake --build "$build_dir" --config Release --parallel "$NPROC" -t database-topn test-applications

    if [ "$3" == "database-build" ]; then
        test_database
        exit 0
    fi

    run_single "$builder" cmake --build "$build_dir" --config Release --parallel "$NPROC" -t knn
    run_single "$builder" cmake --build "$build_dir" --config Release --parallel "$NPROC" -t test

    run_single "$worker" "$build_dir"/test
    test_database

    if [ "$3" == "build" ]; then
        exit 0
    fi
elif [ "$3" == "test" ]; then
    run_single "$worker" "$build_dir"/test
    exit 0
elif [ "$3" == "database-test" ]; then
    test_database
    exit 0
elif [ -n "$3" ]; then
    shift 2
    run_batch "$@"
    exit 0
fi
