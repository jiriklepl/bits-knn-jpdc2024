#!/bin/bash -e

ROOT_DIR="$(realpath "$(dirname "$0")")/.."
NAME="$(basename "$0")"

[ "$0" -ef "$ROOT_DIR/scripts/$NAME" ] || {
    exit 1
}

cd "$ROOT_DIR"

plot_name="${1:-all}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$plot_name" in
    help|-h|--help)
        echo "Usage: $0 [plot-name [plot-arguments ...] | all]" >&2
        echo "application-scaling discovers saved indices when INDEX is omitted (also included in all):" >&2
        echo "  $0 application-scaling [INDEX] [--output-dir PATH]" >&2
        echo "  $0 application-scaling --data-dir data/application-scaling" >&2
        exit 0
        ;;
    all)
        if [ "$#" -gt 0 ]; then
            echo "Plot arguments require a specific plot name." >&2
            exit 1
        fi
        ;;
    *)
        if [ ! -f "scripts/plot-$plot_name.py" ]; then
            echo "Invalid plot name: $plot_name" >&2
            exit 1
        fi
        ;;
esac

python3 -m venv .venv >&2
source .venv/bin/activate >&2
python3 -m pip install -r scripts/requirements.txt >&2

if [ "$plot_name" = "all" ]; then
    find scripts -name "plot-*.py" -exec python3 {} \;
else
    python3 "scripts/plot-$plot_name.py" "$@"
fi
