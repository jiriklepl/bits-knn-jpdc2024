#!/usr/bin/env python3
"""Write the three original plot versions for each application and data size."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from application_scaling_analysis import SIZE_LABELS, load_study, write_outputs


def individual_runs(rows, *, cross_size=False):
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
    if not cross_size:
        expected = {
            (operator, tier)
            for operator in SIZE_LABELS
            for tier in ("small", "middle", "large")
        }
        if (
            len(runs) != 9
            or {(operator, tier) for _, operator, _, tier, _ in runs} != expected
            or any(mode != "fixed-k" for _, _, _, _, mode in runs)
        ):
            raise ValueError(
                "Default plotting requires exactly nine fixed-k cases: each "
                "application must have small, middle, and large data. "
                "Use --cross-size for a legacy or retention study."
            )
        if any(
            Path(path).stem != f"{operator}-{tier}"
            for path, operator, _, tier, _ in runs
        ):
            raise ValueError(
                "Nine-case timing CSV names must be <application>-<size_tier>.csv"
            )
    if len({Path(path).stem for path, *_ in runs}) != len(runs):
        raise ValueError("Individual study runs must have distinct CSV basenames")
    return sorted((path, operator) for path, operator, *_ in runs)


def plot_study(index, output, *, cross_size=False, skip_individual_runs=False):
    rows = load_study(index, annotate=cross_size)
    if output.resolve() == index.parent.resolve():
        raise ValueError("Choose a plot directory separate from raw study outputs")
    runs = individual_runs(rows, cross_size=cross_size)
    if cross_size:
        write_outputs(rows, output)
    individual_output = output / "workloads" if cross_size else output
    if not skip_individual_runs:
        for path, operator in runs:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name(f"plot-{operator}.py")),
                    path,
                    "--output-dir",
                    str(individual_output),
                ],
                check=True,
            )


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
    parser.add_argument(
        "--cross-size",
        action="store_true",
        help=(
            "Also generate the extended cross-size and degree comparisons; "
            "accept legacy studies and put individual plots under workloads/"
        ),
    )
    parser.add_argument(
        "--skip-individual-runs",
        action="store_true",
        help="With --cross-size, omit individual detailed and paper PDFs",
    )
    args = parser.parse_args()
    if args.skip_individual_runs and not args.cross_size:
        parser.error("--skip-individual-runs requires --cross-size")
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
            plot_study(
                index,
                output,
                cross_size=args.cross_size,
                skip_individual_runs=args.skip_individual_runs,
            )
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
