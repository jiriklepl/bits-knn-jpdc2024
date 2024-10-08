#!/bin/bash -e

ROOT_DIR="$(realpath "$(dirname "$0")")/.."
NAME="$(basename "$0")"

[ "$0" -ef "$ROOT_DIR/scripts/$NAME" ] || {
    exit 1
}

cd "$ROOT_DIR"

if [ -n "$1" ]; then
    if [ "$1" = "all" ]; then
        :
    elif [ "$1" = "help" ]; then
        echo "Usage: $0 [plot-name]" >&2
        exit 1
    elif [ ! -f "scripts/plot-$1.py" ]; then
        echo "Invalid plot name: $1" >&2
        exit 1
    fi
fi

python3 -m venv .venv >&2
source .venv/bin/activate >&2
python3 -m pip install -r scripts/requirements.txt >&2

if [ -z "$1" ]; then
    find scripts -name "plot-*.py" -exec python3 {} \;
else
    python3 "scripts/plot-$1.py"
fi
