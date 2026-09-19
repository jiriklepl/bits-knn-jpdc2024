#!/usr/bin/env python3
"""Summarize verified database-topn runs and plot full-operator latency."""

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import median, quantiles
import sys


CONFIG = (
    "dataset_id",
    "rows",
    "k",
    "backend",
    "degree",
    "block_size",
    "items_per_thread",
)
MEASURED_PHASES = {
    "operator",
    "download",
    "transform_isolated",
    "selection_isolated",
    "output_isolated",
}
BACKENDS = ("bits", "bits-sq", "air-topk", "grid-select")  # , "block-select"
LABELS = ("BITS", "BITS (split)", "AIR Top-K", "GridSelect")  # , "BlockSelect"


def summarize(path):
    """Keep inputs, configurations and phases separate; warmups are already excluded."""
    groups = defaultdict(dict)
    dataset_rows = {}
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        required = {*CONFIG, "retention_ratio", "iteration", "phase", "seconds"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"{path}: expected a database-topn timing CSV")
        for line, row in enumerate(reader, 2):
            try:
                for name in (
                    "rows",
                    "k",
                    "degree",
                    "block_size",
                    "items_per_thread",
                    "iteration",
                ):
                    row[name] = int(row[name])
                seconds = float(row["seconds"])
                ratio = float(row["retention_ratio"])
                digest = row["dataset_id"]
                if len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest
                ):
                    raise ValueError("dataset_id must be a manifest SHA-256")
                if row["backend"] not in BACKENDS:
                    continue  # ignore unrecognized backends, e.g., "block-select"
                if not 0 < row["k"] <= min(row["rows"], 2048):
                    raise ValueError("invalid row count or k")
                if (
                    not 0 < row["degree"] <= row["rows"]
                    or min(row["block_size"], row["items_per_thread"]) < 0
                ):
                    raise ValueError("invalid selector configuration")
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("timings must be positive and finite")
                if not math.isclose(ratio, row["k"] / row["rows"], rel_tol=1e-9):
                    raise ValueError("retention ratio does not match k / rows")
                if dataset_rows.setdefault(digest, row["rows"]) != row["rows"]:
                    raise ValueError("one dataset_id has different row counts")
                phase, iteration = row["phase"], row["iteration"]
                if phase == "upload_shared":
                    if iteration != -1:
                        raise ValueError("upload_shared must use iteration -1")
                elif phase not in MEASURED_PHASES or iteration < 0:
                    raise ValueError("unknown phase or invalid measured iteration")
                key = tuple(row[name] for name in CONFIG) + (phase,)
                if iteration in groups[key]:
                    raise ValueError(
                        "duplicate timing for the same configuration and iteration"
                    )
                groups[key][iteration] = seconds * 1000
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    if not groups:
        raise ValueError(f"{path}: no measurements")

    # Truncated runs must not silently produce a partial comparison.
    configurations = {key[:-1] for key in groups}
    sample_counts = set()
    for config in configurations:
        operator = groups.get(config + ("operator",), {})
        iterations = set(operator)
        if not iterations or iterations != set(range(len(iterations))):
            raise ValueError(
                f"{path}: measured iterations must be contiguous from zero"
            )
        for phase in MEASURED_PHASES:
            if set(groups.get(config + (phase,), {})) != iterations:
                raise ValueError(f"{path}: incomplete measured phases for {config}")
        if set(groups.get(config + ("upload_shared",), {})) != {-1}:
            raise ValueError(f"{path}: missing shared upload for {config}")
        sample_counts.add(len(iterations))
    if len(sample_counts) != 1:
        raise ValueError(f"{path}: configurations have different repetition counts")

    summary = []
    for key, samples in sorted(groups.items()):
        values = list(samples.values())
        q1, _, q3 = (
            quantiles(values, n=4, method="inclusive")
            if len(values) > 1
            else values * 3
        )
        summary.append(
            dict(zip(CONFIG + ("phase",), key))
            | {
                "samples": len(values),
                "median_ms": median(values),
                "p25_ms": q1,
                "p75_ms": q3,
            }
        )
    return summary


def plot(summary, path, output_dir):
    import matplotlib.pyplot as plt
    import utils

    datasets = defaultdict(list)
    for row in summary:
        if row["phase"] == "operator":
            datasets[row["dataset_id"], row["rows"]].append(row)
    for (digest, rows), values in datasets.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        series = defaultdict(list)
        for row in values:
            # BlockSelect's queue length is automatically chosen from k.
            items = (
                None if row["backend"] == "block-select" else row["items_per_thread"]
            )
            series[row["backend"], row["degree"], row["block_size"], items].append(row)
        ordered = sorted(
            series.items(), key=lambda pair: (BACKENDS.index(pair[0][0]), pair[0][1:])
        )
        for (backend, degree, block, items), points in ordered:
            points.sort(key=lambda row: row["k"])
            if len({row["k"] for row in points}) != len(points):
                raise ValueError(
                    f"{path}: multiple configurations at the same plotted point"
                )
            index = BACKENDS.index(backend)
            label = LABELS[index]
            if backend in ("bits", "bits-sq"):
                label += f" (block={block}, batch={items}"
                label += f", degree={degree})" if backend == "bits-sq" else ")"
            center = [row["median_ms"] for row in points]
            ax.errorbar(
                [row["k"] for row in points],
                center,
                yerr=[
                    [row["median_ms"] - row["p25_ms"] for row in points],
                    [row["p75_ms"] - row["median_ms"] for row in points],
                ],
                label=label,
                color=utils.COLORS[index],
                marker=utils.SHAPES[index],
                capsize=3,
                linewidth=1.5,
            )
        ks = sorted({row["k"] for row in values})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, labels=[str(k) for k in ks])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Selected rows (k)")
        ax.set_ylabel("Full operator latency [ms]")
        ax.set_title(
            f"{rows:,} rows; input {digest[:10]}\nMedian and interquartile range"
        )
        ax.grid(alpha=0.4, linestyle="--")
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
        )
        fig.tight_layout()
        suffix = f"-{digest[:10]}" if len(datasets) > 1 else ""
        fig.savefig(output_dir / f"{path.stem}{suffix}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    try:
        files = args.files
        if not files:
            files = []
            for path in sorted(Path("data").glob("database-topn-*-*.csv")):
                if path.stat().st_size == 0:
                    print(f"Skipping empty run: {path}", file=sys.stderr)
                else:
                    files.append(path)
        if not files:
            print("No database-topn timing files found.", file=sys.stderr)
            return
        for path in files:
            summary = summarize(path)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            plot(summary, path, args.output_dir)
            with (args.output_dir / f"{path.stem}.csv").open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
