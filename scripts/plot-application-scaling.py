#!/usr/bin/env python3
"""Write detailed cases and combined three-application paper plots per data size."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from application_scaling_analysis import OPERATORS, load_study
from application_plotting import plot_combined_paper


def individual_runs(rows):
    """Require the complete nine-case layout before starting any default plots."""
    runs = {
        (
            row["source_csv"],
            row["operator"],
            row["workload_id"],
            row["size_tier"],
            row["mode"],
        )
        for row in rows
    }
    expected = {
        (operator, tier)
        for operator in OPERATORS
        for tier in ("small", "middle", "large")
    }
    if (
        len(runs) != 9
        or {(operator, tier) for _, operator, _, tier, _ in runs} != expected
        or any(mode != "fixed-k" for _, _, _, _, mode in runs)
    ):
        raise ValueError(
            "Plotting requires exactly nine fixed-k cases: each "
            "application must have small, middle, and large data."
        )
    if any(
        Path(path).stem != f"{operator}-{tier}" for path, operator, _, tier, _ in runs
    ):
        raise ValueError(
            "Nine-case timing CSV names must be <application>-<size_tier>.csv"
        )
    if len({Path(path).stem for path, *_ in runs}) != len(runs):
        raise ValueError("Individual study runs must have distinct CSV basenames")
    return sorted((path, operator) for path, operator, *_ in runs)


def plot_study(index, output):
    rows = load_study(index)
    if output.resolve() == index.parent.resolve():
        raise ValueError("Choose a plot directory separate from raw study outputs")
    runs = individual_runs(rows)
    for path, operator in runs:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(f"plot-{operator}.py")),
                path,
                "--detailed-only",
                "--output-dir",
                str(output),
            ],
            check=True,
        )
    plot_combined_paper(rows, output)
    # These generated files are superseded by the combined paper figures.
    for path, _ in runs:
        for suffix in (
            "-paper.pdf",
            "-paper-global.pdf",
            "-paper-configs.csv",
            "-paper-global-configs.csv",
        ):
            (output / f"{Path(path).stem}{suffix}").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index",
        type=Path,
        nargs="?",
        help="Plot this study, or recursively discover indices under --data-dir",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/application-scaling"),
        help="Discovery root without an index (default: data/application-scaling)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Plot directory for an explicit index (default: plots/<study>); "
            "output root for discovery (default: plots/application-scaling), "
            "preserving study subdirectories"
        ),
    )
    args = parser.parse_args()
    indices = [args.index] if args.index else sorted(args.data_dir.rglob("index.json"))
    if not indices:
        print(f"No study indices found under {args.data_dir}", file=sys.stderr)
        return
    failed = False
    for index in indices:
        try:
            if args.index:
                output = args.output_dir or Path("plots") / index.parent.name
            else:
                runs = json.loads(index.read_text())["runs"]
                if runs and any(run["status"] != "complete" for run in runs):
                    print(f"Skipping incomplete study: {index}", file=sys.stderr)
                    continue
                output = (args.output_dir or Path("plots/application-scaling")) / (
                    index.parent.relative_to(args.data_dir)
                )
            print(f"Plotting {index} -> {output}", file=sys.stderr)
            plot_study(index, output)
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.CalledProcessError,
        ) as error:
            print(f"{index}: {error}", file=sys.stderr)
            failed = True
    if failed:
        parser.exit(1)


if __name__ == "__main__":
    main()
