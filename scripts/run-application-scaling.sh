#!/bin/bash
#SBATCH -o data/application-scaling-%N-%j.log
#SBATCH -e data/application-scaling-%N-%j.err
#SBATCH --gpus 1
#SBATCH -p gpu-short
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --exclusive

set -eo pipefail

root_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}"
suite="$root_dir/data/application-inputs/scaling/suite.json"
output_dir=""
resume=false
runner_options=()

usage() {
    cat <<'EOF'
Usage: scripts/run-application-scaling.sh [--suite FILE] [--output-dir DIR] [RUNNER_OPTIONS]

Run all nine application/size cases using the build wrapper's selected binaries.
The default suite is data/application-inputs/scaling/suite.json.
Results default to data/application-scaling/WORKER-JOBID (or a unique local ID).
Use --resume --output-dir DIR to continue an interrupted matching run.
Other options, such as --ks, --degrees, --warmup and --repeat, reach the runner.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --suite|--output-dir)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
                printf '%s requires a path\n' "$1" >&2
                exit 1
            fi
            if [ "$1" = "--suite" ]; then
                suite="$2"
            else
                output_dir="$2"
            fi
            shift 2
            ;;
        --suite=*|--output-dir=*)
            if [ -z "${1#*=}" ]; then
                printf '%s requires a path\n' "${1%%=*}" >&2
                exit 1
            fi
            if [[ "$1" == --suite=* ]]; then
                suite="${1#*=}"
            else
                output_dir="${1#*=}"
            fi
            shift
            ;;
        --resume)
            resume=true
            runner_options+=("$1")
            shift
            ;;
        *)
            runner_options+=("$1")
            shift
            ;;
    esac
done

if $resume && [ -z "$output_dir" ]; then
    echo "--resume requires --output-dir pointing to the interrupted run" >&2
    exit 1
fi

worker="${worker:-$(hostname)}"
# shellcheck source=scripts/config.sh
. "$root_dir/scripts/config.sh"

if [ -z "$output_dir" ]; then
    application_run_id="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
    output_dir="$root_dir/data/application-scaling/$worker-$application_run_id"
fi

printf 'Application study directory: %s\n' "$output_dir" >&2
exec python3 "$root_dir/scripts/run-application-scaling.py" "$suite" \
    --build-dir "$build_dir" --output-dir "$output_dir" "${runner_options[@]}"
