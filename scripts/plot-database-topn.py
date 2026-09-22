#!/usr/bin/env python3
"""Summarize database-topn runs and plot full-operator speedup against AIR Top-K."""

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import median, quantiles
import sys

from application_plotting import (
    add_speedup_series,
    annotate_paper_selection,
    configuration_pages,
    fit_speedup_axes,
    write_configuration_csv,
    select_paper_rows,
)


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
BACKENDS = (
    "bits-prefetch",
    "bits-sq",
    "air-topk",
    "grid-select",
    "bits",
    "block-select",
)
LABELS = (
    "bits",
    "bits (split)",
    "AIR Top-K",
    "GridSelect",
    "bits (no prefetch)",
    "BlockSelect",
)
WORKLOAD = ("dataset_id", "rows", "k", "phase")


def summarize(path):
    """Keep inputs, configurations and phases separate; warmups are already excluded."""
    groups = defaultdict(dict)
    dataset_rows = {}
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        required = {*CONFIG, "retention_ratio", "iteration", "phase", "seconds"}
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(required)
            or set(reader.fieldnames) != required
        ):
            raise ValueError(f"{path}: expected a database-topn timing CSV")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV columns")
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
                    raise ValueError("unknown backend")
                if not 0 < row["k"] <= min(row["rows"], 2048):
                    raise ValueError("invalid row count or k")
                if (
                    not 0 < row["degree"] <= row["rows"]
                    or min(row["block_size"], row["items_per_thread"]) < 0
                ):
                    raise ValueError("invalid selector configuration")
                if not math.isfinite(seconds * 1000) or seconds <= 0:
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
    baselines = defaultdict(list)
    for row in summary:
        if row["backend"] == "air-topk":
            baselines[tuple(row[name] for name in WORKLOAD)].append(row)
    for row in summary:
        matches = baselines[tuple(row[name] for name in WORKLOAD)]
        row["speedup_vs_air"] = (
            matches[0]["median_ms"] / row["median_ms"] if len(matches) == 1 else ""
        )
    return summary


def validate_plot_summary(summary, path, paper=False, *, selection="per-k"):
    """Require one AIR baseline and unambiguous plotted points before rendering."""
    datasets = defaultdict(list)
    for row in summary:
        if row["phase"] != "operator" or (paper and row["backend"] == "block-select"):
            continue
        if row["speedup_vs_air"] == "":
            raise ValueError(
                f"{path}: plotting requires a unique matching AIR Top-K baseline"
            )
        datasets[row["dataset_id"], row["rows"]].append(row)
    for values in datasets.values():
        seen = set()
        for row in values:
            names = ("backend", "degree", "block_size", "items_per_thread", "k")
            key = tuple(row[name] for name in names)
            if key in seen:
                raise ValueError(
                    f"{path}: multiple configurations for the same "
                    f"{'paper ' if paper else ''}backend+k point"
                )
            seen.add(key)
    if paper:
        return {
            key: select_paper_rows(values, path, selection=selection)
            for key, values in datasets.items()
        }
    return datasets


def plot(summary, path, output_dir, paper=False, *, selection="per-k"):
    datasets = validate_plot_summary(summary, path, paper, selection=selection)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import utils

    markers = tuple(dict.fromkeys(utils.SHAPES))
    suffix = ("-paper-global" if selection == "global" else "-paper") if paper else ""
    with PdfPages(output_dir / f"{path.stem}{suffix}.pdf") as pdf:
        for (digest, rows), degree_page, values in configuration_pages(
            datasets, paper=paper
        ):
            series = defaultdict(list)
            for row in values:
                key = (
                    (row["backend"],)
                    if paper
                    else tuple(
                        row[name]
                        for name in (
                            "backend",
                            "degree",
                            "block_size",
                            "items_per_thread",
                        )
                    )
                )
                series[key].append(row)
            # Preserve the plotting area as sweep legends gain more rows.
            legend_rows = math.ceil(len(series) / 2)
            height = 5 + (0 if paper else 0.18 * max(0, legend_rows - 5))
            fig, ax = plt.subplots(figsize=(7 if paper else 8, height))
            ordered = sorted(
                series.items(),
                key=lambda pair: (BACKENDS.index(pair[0][0]), pair[0][1:]),
            )
            variants = defaultdict(int)
            handles, labels = [], []
            for config, points in ordered:
                points.sort(key=lambda row: row["k"])
                backend = config[0]
                index = BACKENDS.index(backend)
                variant = variants[backend]
                variants[backend] += 1
                label = LABELS[index]
                if not paper:
                    _, degree, block, items = config
                    label += f" (degree={degree}, block={block}, items={items})"
                handle = add_speedup_series(
                    ax,
                    points,
                    label=label,
                    color=utils.COLORS[index],
                    marker=markers[(index + variant // 4) % len(markers)],
                    linestyle=("-", "--", "-.", ":")[variant % 4],
                )
                handles.append(handle)
                labels.append(label)
            ks = sorted({row["k"] for row in values})
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks, labels=[str(k) for k in ks])
            ax.set_xlabel("Selected rows (k)")
            ax.set_ylabel("Full operator\nspeedup vs AIR Top-K [×]")
            if not paper:
                split_detail = (
                    f"; split degree={degree_page}" if degree_page is not None else ""
                )
                ax.set_title(
                    f"{rows:,} rows; input {digest[:10]}{split_detail}\n"
                    "AIR median / backend median; >1 is faster\n"
                    "Bars: backend IQR; AIR median fixed",
                    fontsize=11,
                )
            ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
            ax.grid(alpha=0.4, linestyle="--")
            fit_speedup_axes([ax])
            ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize=8,
                ncol=2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    if paper:
        write_configuration_csv(
            output_dir / f"{path.stem}{suffix}-configs.csv",
            [
                row | {"source_csv": str(path)}
                for values in datasets.values()
                for row in values
            ],
            selection=selection,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--detailed-only",
        action="store_true",
        help="Paper figures are generated by the combined study plotter",
    )
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
        inputs = {path.resolve() for path in files}
        destinations = set()
        for path in files:
            for suffix in (
                ".csv",
                ".pdf",
                "-paper.pdf",
                "-paper-global.pdf",
                "-paper-configs.csv",
                "-paper-global-configs.csv",
            ):
                destination = args.output_dir / f"{path.stem}{suffix}"
                resolved = destination.resolve()
                if resolved in inputs:
                    raise ValueError(
                        "Analysis output would overwrite an input timing file: "
                        f"{destination}"
                    )
                if resolved in destinations:
                    raise ValueError(f"Analysis output paths collide: {destination}")
                destinations.add(resolved)
        summaries = [(path, summarize(path)) for path in files]
        for path, summary in summaries:
            validate_plot_summary(summary, path)
            validate_plot_summary(summary, path, paper=True)
            validate_plot_summary(summary, path, paper=True, selection="global")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, summary in summaries:
            plot(summary, path, args.output_dir, paper=False)
            if not args.detailed_only:
                plot(summary, path, args.output_dir, paper=True)
                plot(summary, path, args.output_dir, paper=True, selection="global")
            with (args.output_dir / f"{path.stem}.csv").open("w", newline="") as target:
                annotated = annotate_paper_selection(
                    summary, ("dataset_id", "rows"), path
                )
                writer = csv.DictWriter(target, fieldnames=list(annotated[0]))
                writer.writeheader()
                writer.writerows(annotated)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
