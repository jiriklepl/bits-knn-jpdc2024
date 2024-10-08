#!/bin/bash -ex

export worker=${1:-$(hostname)}
builder=${builder:-"$worker"}
export CUDA_ARCHITECTURES=${2:-"80"}
export build_dir="build-$worker"

run_batch() {
    # roughly simulate the behavior of sbatch
    jobid="$(date +%s)"
    default_name=$(basename "$1" | sed 's/\.[^.]\+$//;s/run-/data\//')

    out_file=$(awk '/#SBATCH -o/{print $3; nodefault=1; exit 0 }END{ if (!nodefault) print "'"$default_name-%N-%j.csv"'"}' "$1" | sed 's/%N/'"$worker"'/;s/%j/'"$jobid"'/')
    err_file=$(awk '/#SBATCH -e/{print $3; nodefault=1; exit 0 }END{ if (!nodefault) print "'"$default_name-%N-%j.err"'"}' "$1" | sed 's/%N/'"$worker"'/;s/%j/'"$jobid"'/')

    dirname "$out_file" | xargs mkdir -p
    dirname "$err_file" | xargs mkdir -p

    "$@" >"$out_file" 2>"$err_file" || true
}

run_single() {
    "$@"
}

. ./scripts/executor.sh
