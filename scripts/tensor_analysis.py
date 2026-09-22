"""Validate and summarize tensor-application timings without pooling runs."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import median, quantiles
import sys

from application_plotting import (
    annotate_paper_selection,
    configuration_pages,
    write_configuration_csv,
    select_paper_rows,
)

OPERATORS = ("token-sampling", "gradient-compression")
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
WORKLOAD = ("operator", "dataset_id", "rows", "batch_size", "k", "temperature", "seed")
CONFIG = WORKLOAD + ("backend", "degree", "block_size", "items_per_thread")
MEASURED_PHASES = {
    "operator",
    "download",
    "transform_isolated",
    "selection_isolated",
    "output_isolated",
}
CSV_FIELDS = {*CONFIG, "retention_ratio", "iteration", "phase", "seconds"}
INTEGER_FIELDS = (
    "rows",
    "batch_size",
    "k",
    "seed",
    "degree",
    "block_size",
    "items_per_thread",
    "iteration",
)


def summarize(path, operator):
    """Keep phases/configurations separate; native CSVs already exclude warmups."""
    if operator not in OPERATORS:
        raise ValueError(f"Unknown tensor operator: {operator}")
    path = Path(path)
    groups = defaultdict(dict)
    dataset_shapes = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(CSV_FIELDS)
            or set(reader.fieldnames) != CSV_FIELDS
        ):
            raise ValueError(f"{path}: expected a {operator} timing CSV")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV columns")
                for name in INTEGER_FIELDS:
                    row[name] = int(row[name])
                row["temperature"] = float(row["temperature"])
                seconds = float(row["seconds"])
                ratio = float(row["retention_ratio"])
                if row["operator"] != operator:
                    raise ValueError(f"expected operator {operator}")
                digest = row["dataset_id"]
                if len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest
                ):
                    raise ValueError("dataset_id must be a manifest SHA-256")
                if row["backend"] not in BACKENDS:
                    raise ValueError("unknown backend")
                if (
                    not 0 < row["rows"] <= 2**31 - 1
                    or not 0 < row["batch_size"] <= 2**31 - 1
                ):
                    raise ValueError("invalid input shape")
                if not 0 < row["k"] <= min(row["rows"], 2048):
                    raise ValueError("invalid k")
                if (
                    not 0 < row["degree"] <= row["rows"]
                    or min(row["block_size"], row["items_per_thread"]) < 0
                ):
                    raise ValueError("invalid selector configuration")
                temperature, seed = row["temperature"], row["seed"]
                if not math.isfinite(temperature) or not 0 <= seed < 2**64:
                    raise ValueError("invalid temperature or seed")
                if operator == "token-sampling" and temperature <= 0:
                    raise ValueError("sampling temperature must be positive")
                if operator == "gradient-compression" and (
                    row["batch_size"] != 1 or temperature != 0 or seed != 0
                ):
                    raise ValueError(
                        "gradient compression requires batch_size=1, "
                        "temperature=0, seed=0"
                    )
                if not math.isfinite(seconds * 1000) or seconds <= 0:
                    raise ValueError("timings must be positive and finite")
                if not math.isclose(ratio, row["k"] / row["rows"], rel_tol=1e-9):
                    raise ValueError("retention ratio does not match k / rows")
                shape = row["rows"], row["batch_size"]
                if dataset_shapes.setdefault(digest, shape) != shape:
                    raise ValueError("one dataset_id has different input shapes")
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
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    if not groups:
        raise ValueError(f"{path}: no measurements")

    sample_counts = set()
    for config in {key[:-1] for key in groups}:
        iterations = set(groups.get(config + ("operator",), {}))
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
        row = dict(zip(CONFIG + ("phase",), key))
        summary.append(
            row
            | {
                "source_file": str(path),
                "retention_ratio": row["k"] / row["rows"],
                "samples": len(values),
                "median_ms": median(values),
                "p25_ms": q1,
                "p75_ms": q3,
            }
        )

    # A baseline is a unique AIR configuration for this exact workload and phase.
    # Multiple AIR configurations are retained, but none is selected as the best.
    baselines = defaultdict(list)
    for row in summary:
        if row["backend"] == "air-topk":
            baselines[tuple(row[name] for name in WORKLOAD) + (row["phase"],)].append(
                row
            )
    for row in summary:
        matches = baselines[tuple(row[name] for name in WORKLOAD) + (row["phase"],)]
        row["speedup_vs_air"] = (
            matches[0]["median_ms"] / row["median_ms"] if len(matches) == 1 else ""
        )
    return summary


def discover_files(operator, data_dir=Path("data")):
    """Ignore only zero-byte failed jobs during automatic discovery."""
    files = []
    for path in sorted(Path(data_dir).glob(f"{operator}-*-*.csv")):
        if path.stat().st_size == 0:
            print(f"Skipping empty run: {path}", file=sys.stderr)
        else:
            files.append(path)
    return files


def plot_pages(summary, path, paper=False, *, selection="per-k"):
    """Validate AIR comparisons and unambiguous paper curves before writing files."""
    pages = defaultdict(list)
    seen = set()
    for row in summary:
        if row["phase"] not in ("operator", "selection_isolated"):
            continue
        if row["speedup_vs_air"] == "":
            raise ValueError(
                f"{path}: plotting requires a unique AIR Top-K baseline for "
                f"each workload, k and phase ({row['phase']}, k={row['k']})"
            )
        key = tuple(row[name] for name in WORKLOAD if name != "k")
        pages[key].append(row)
        point = tuple(row[name] for name in CONFIG) + (row["phase"],)
        if point in seen:
            raise ValueError(
                f"{path}: duplicate configuration for the same backend+k+phase point"
            )
        seen.add(point)
    if paper:
        return {
            key: select_paper_rows(rows, path, selection=selection)
            for key, rows in pages.items()
        }
    return pages


def plot(summary, path, output_dir, *, detailed_only=False):
    """Write detailed, per-k and global paper PDFs for each input/settings group."""
    pages = plot_pages(summary, path)
    paper_pages = plot_pages(summary, path, paper=True)
    global_paper_pages = plot_pages(summary, path, paper=True, selection="global")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from application_plotting import add_speedup_series, fit_speedup_axes
    import utils

    markers = tuple(dict.fromkeys(utils.SHAPES))
    for suffix, paper, selected_pages in (
        ("", False, pages),
        ("-paper", True, paper_pages),
        ("-paper-global", True, global_paper_pages),
    ):
        if paper and detailed_only:
            continue
        with PdfPages(output_dir / f"{path.stem}{suffix}.pdf") as pdf:
            for key, degree_page, points in configuration_pages(
                selected_pages, paper=paper
            ):
                operator, digest, rows, batch, temperature, seed = key
                visible = [
                    row
                    for row in points
                    if not paper or row["backend"] != "block-select"
                ]
                configurations = {
                    tuple(
                        row[name]
                        for name in (
                            "backend",
                            "degree",
                            "block_size",
                            "items_per_thread",
                        )
                    )
                    for row in visible
                    if row["phase"] == "operator"
                }
                legend_rows = math.ceil(len(configurations) / 2)
                legend_height = 0.18 * legend_rows + 0.15
                height = 5 if paper else 5 + legend_height
                fig, axes = plt.subplots(1, 2, figsize=(12, height), sharey=True)
                handles, labels = [], []
                for ax, phase, title in zip(
                    axes,
                    ("operator", "selection_isolated"),
                    ("Full operator", "Isolated selection"),
                ):
                    series = defaultdict(list)
                    for row in visible:
                        if row["phase"] == phase:
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
                    ordered = sorted(
                        series.items(),
                        key=lambda pair: (BACKENDS.index(pair[0][0]), pair[0][1:]),
                    )
                    configuration_counts = defaultdict(int)
                    for key, values in ordered:
                        backend = key[0]
                        values.sort(key=lambda row: row["k"])
                        index = BACKENDS.index(backend)
                        variant = configuration_counts[backend]
                        configuration_counts[backend] += 1
                        label = LABELS[index]
                        if not paper:
                            _, degree, block, items = key
                            label += f" (degree={degree}, block={block}, items={items})"
                        handle = add_speedup_series(
                            ax,
                            values,
                            label=label,
                            color=utils.COLORS[index],
                            marker=markers[(index + variant // 4) % len(markers)],
                            linestyle=("-", "--", "-.", ":")[variant % 4],
                        )
                        if phase == "operator":
                            handles.append(handle)
                            labels.append(label)
                    ks = sorted({row["k"] for row in visible if row["phase"] == phase})
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(ks, labels=[str(k) for k in ks])
                    ax.set_xlabel("Selected candidates (k)")
                    if paper:
                        ax.set_ylabel(f"{title}\nspeedup vs AIR Top-K [×]")
                    else:
                        ax.set_title(title)
                    ax.grid(alpha=0.4, linestyle="--")
                    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
                fit_speedup_axes(axes)
                if not paper:
                    axes[0].set_ylabel("Speedup vs AIR Top-K [×]")
                    detail = f"{rows:,} candidates; batch={batch}; input {digest[:10]}"
                    if degree_page is not None:
                        detail += f"; split degree={degree_page}"
                    if operator == "token-sampling":
                        detail += f"; temperature={temperature!r}; seed={seed}"
                    fig.suptitle(
                        f"{operator}: {detail}\n{path.stem} — "
                        "AIR median / backend median; >1 is faster\n"
                        "Bars: backend IQR; AIR median fixed",
                        fontsize=11,
                    )
                columns = 2 if suffix == "-paper-global" or not paper else 4
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=columns,
                    frameon=False,
                    fontsize=11 if paper else 8,
                )
                fig.tight_layout(
                    rect=(
                        0,
                        0.045 * math.ceil(len(labels) / columns)
                        if paper
                        else legend_height / height,
                        1,
                        1,
                    )
                )
                pdf.savefig(fig)
                plt.close(fig)
        if paper:
            write_configuration_csv(
                output_dir / f"{path.stem}{suffix}-configs.csv",
                [
                    row | {"source_csv": str(path)}
                    for values in selected_pages.values()
                    for row in values
                ],
                selection="global" if suffix == "-paper-global" else "per-k",
            )


def main(operator):
    parser = argparse.ArgumentParser(
        description=(
            f"Write detailed, per-k paper and global paper AIR Top-K comparisons "
            f"for {operator}."
        )
    )
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--detailed-only",
        action="store_true",
        help="Paper figures are generated by the combined study plotter",
    )
    args = parser.parse_args()
    try:
        files = args.files or discover_files(operator)
        if not files:
            print(f"No {operator} timing files found.", file=sys.stderr)
            return
        if len({path.stem for path in files}) != len(files):
            raise ValueError(
                "Input files must have distinct basenames; "
                "analyze matching names separately"
            )
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
                if destination.resolve() in inputs:
                    raise ValueError(
                        "Analysis output would overwrite an input timing file: "
                        f"{destination}"
                    )
                if destination.resolve() in destinations:
                    raise ValueError(
                        f"Analysis output filenames collide: {destination}"
                    )
                destinations.add(destination.resolve())
        # Validate all inputs before producing any comparison from this invocation.
        summaries = [(path, summarize(path, operator)) for path in files]
        for path, summary in summaries:
            plot_pages(summary, path)
            plot_pages(summary, path, paper=True)
            plot_pages(summary, path, paper=True, selection="global")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, summary in summaries:
            plot(summary, path, args.output_dir, detailed_only=args.detailed_only)
            with (args.output_dir / f"{path.stem}.csv").open("w", newline="") as target:
                annotated = annotate_paper_selection(
                    summary, tuple(name for name in WORKLOAD if name != "k"), path
                )
                writer = csv.DictWriter(target, fieldnames=list(annotated[0]))
                writer.writeheader()
                writer.writerows(annotated)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
