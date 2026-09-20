"""Compare application sizes within one verified scaling study."""

import csv
from collections import defaultdict
import importlib.util
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from application_inputs import sha256_file
from application_plotting import select_paper_rows, speedup_errors
import tensor_analysis


BACKENDS = ("bits-prefetch", "bits-sq", "air-topk", "grid-select")
LABELS = ("BITS", "BITS (split)", "AIR Top-K", "GridSelect")
POINT = ("workload_id", "scenario")
CONFIG = ("degree", "block_size", "items_per_thread")
SIZE_LABELS = {
    "database-topn": "Input rows",
    "token-sampling": "Sequences in the batch",
    "gradient-compression": "Gradient elements",
}


def expected_configurations(settings, ks):
    """Match the study matrix and the native comparison backends' reported settings."""
    expected = set()
    for k in ks:
        expected.update(
            (backend, k, degree, block, items)
            for backend, degrees in (
                ("bits-prefetch", (1,)),
                ("bits-sq", settings["degrees"]),
            )
            for degree in degrees
            for block in settings["blocks"]
            for items in settings["items"]
        )
        expected.update((("air-topk", k, 1, 512, 0), ("grid-select", k, 1, 0, 0)))
        if k in (32, 64, 128, 256, 512, 1024):
            queue = 2 if k == 32 else 3 if k <= 128 else 4 if k == 256 else 8
            expected.add(("block-select", k, 1, 128, queue))
    return expected


def validate_run(summary, run, workload, settings, path):
    """Reject shape, scenario, or matrix mismatches even when raw hashes agree."""
    operator = workload["operator"]
    rows = {
        "database-topn": workload.get("rows"),
        "token-sampling": workload.get("vocabulary_size"),
        "gradient-compression": workload.get("elements"),
    }[operator]
    batch = workload["batch_size"] if operator == "token-sampling" else 1
    unit = {
        "database-topn": "rows",
        "token-sampling": "batch",
        "gradient-compression": "elements",
    }[operator]
    if (workload["size"], workload["size_unit"], workload["score_bytes"]) != (
        batch if operator == "token-sampling" else rows,
        unit,
        4 * rows * batch,
    ):
        raise ValueError(f"Workload dimensions are inconsistent: {workload['id']}")
    ks = settings["ks"]
    if run["mode"] == "retention":
        k = max(
            1,
            int(
                (Decimal(str(run["requested_retention"])) * rows).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            ),
        )
        ks = [k]
        if not math.isclose(run["actual_retention"], k / rows, rel_tol=1e-12):
            raise ValueError(f"Actual retention disagrees with requested ratio: {path}")
    if set(run["ks"]) != set(ks) or len(run["ks"]) != len(ks):
        raise ValueError(f"Planned k values disagree with study settings: {path}")
    if {row["dataset_id"] for row in summary} != {workload["dataset_id"]}:
        raise ValueError(f"Timing dataset does not match workload: {path}")
    if {row["k"] for row in summary} != set(ks):
        raise ValueError(f"Timing k values do not match planned run: {path}")
    for row in summary:
        if (row["rows"], row.get("batch_size", 1)) != (rows, batch):
            raise ValueError(f"Timing shape does not match workload: {path}")
        if operator == "token-sampling" and (row["temperature"], row["seed"]) != (
            1,
            42,
        ):
            raise ValueError(f"Sampling settings differ from the scaling study: {path}")
        if row["samples"] != (
            1 if row["phase"] == "upload_shared" else settings["repeat"]
        ):
            raise ValueError(f"Timing repetition count differs from study: {path}")
        if row["speedup_vs_air"] == "":
            raise ValueError(f"Missing unique AIR baseline: {path}")
    observed = {
        tuple(row[name] for name in ("backend", "k") + CONFIG)
        for row in summary
        if row["phase"] == "operator"
    }
    if observed != expected_configurations(settings, ks):
        raise ValueError(
            f"Timing configurations do not cover the planned sweep: {path}"
        )
    if run["configurations"] != len(observed):
        raise ValueError(
            f"Configuration count differs from completed study run: {path}"
        )


def load_study(index_path, *, annotate=True):
    """Check provenance and completeness before combining any timing files."""
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text())
    if index.get("version") != 1:
        raise ValueError(f"{index_path}: unsupported study version")
    settings = index["settings"]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if digest != index["settings_sha256"]:
        raise ValueError("Scaling settings checksum mismatch")
    for field in ("ks", "degrees", "blocks", "items"):
        values = settings[field]
        if (
            not values
            or len(values) != len(set(values))
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in values
            )
        ):
            raise ValueError(f"Invalid scaling settings: {field}")
    if (
        not isinstance(settings["repeat"], int)
        or isinstance(settings["repeat"], bool)
        or settings["repeat"] < 1
    ):
        raise ValueError("Invalid scaling repetition count")
    workloads = {item["id"]: item for item in index["workloads"]}
    if len(workloads) != len(index["workloads"]):
        raise ValueError("Duplicate workload ID in scaling index")
    if not index.get("runs"):
        raise ValueError("Scaling study has no runs")
    spec = importlib.util.spec_from_file_location(
        "scaling_database_analysis", Path(__file__).with_name("plot-database-topn.py")
    )
    database = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database)
    result, seen, sources = [], set(), set()
    for run in index["runs"]:
        if run["status"] != "complete":
            raise ValueError(
                f"Incomplete study: {run['workload_id']} / {run['mode']} "
                f"is {run['status']}: {run.get('reason', '')}"
            )
        workload = workloads[run["workload_id"]]
        operator = workload["operator"]
        mode = run["mode"]
        ratio = run.get("requested_retention")
        if (
            mode not in ("fixed-k", "retention")
            or (mode == "fixed-k" and ratio is not None)
            or (
                mode == "retention"
                and (
                    operator != "gradient-compression"
                    or not math.isfinite(ratio)
                    or not 0 < ratio <= 1
                )
            )
        ):
            raise ValueError("Invalid scaling comparison mode or retention ratio")
        key = (workload["id"], mode, ratio)
        if key in seen:
            raise ValueError(f"Duplicate study run: {key}")
        seen.add(key)
        path = (index_path.parent / run["csv"]).resolve()
        if path in sources:
            raise ValueError(f"Multiple study runs refer to one timing CSV: {path}")
        sources.add(path)
        if sha256_file(path) != run["csv_sha256"]:
            raise ValueError(f"Timing CSV checksum mismatch: {path}")
        summary = (
            database.summarize(path)
            if operator == "database-topn"
            else tensor_analysis.summarize(path, operator)
        )
        validate_run(summary, run, workload, settings, path)
        for row in summary:
            size = (
                row.get("batch_size", 1)
                if operator == "token-sampling"
                else row["rows"]
            )
            result.append(
                row
                | {
                    "operator": operator,
                    "workload_id": workload["id"],
                    "workload_label": workload["label"],
                    "size_tier": workload.get("size_tier", ""),
                    "size": size,
                    "size_unit": workload["size_unit"],
                    "score_bytes": workload["score_bytes"],
                    "mode": mode,
                    "requested_retention": ratio if ratio is not None else "",
                    "scenario": row["k"] if mode == "fixed-k" else ratio,
                    "retention_ratio": row["k"] / row["rows"],
                    "throughput_mvalues_per_s": row["rows"]
                    * row.get("batch_size", 1)
                    / (1000 * row["median_ms"]),
                    "source_csv": str(path),
                }
            )
    # A missing run entry must not silently turn a partial study into a full one.
    expected = {(workload_id, "fixed-k", None) for workload_id in workloads}
    expected.update(
        (workload_id, "retention", ratio)
        for workload_id, workload in workloads.items()
        if workload["operator"] == "gradient-compression"
        for ratio in index["settings"].get("retention_ratios", [])
    )
    if seen != expected:
        raise ValueError(
            "Study index does not cover every workload and retention ratio"
        )
    return annotate_choices(result, index_path) if annotate else result


def annotate_choices(rows, path):
    """Keep all measurements and choose configurations over sizes and scenarios."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["operator"], row["mode"]].append(row)
    flags = {
        "scaling_selected": set(),
        "scaling_global_selected": set(),
        "degree_selected": set(),
    }
    for values in groups.values():
        for name, selection, fields in (
            ("scaling_selected", "per-k", POINT),
            ("scaling_global_selected", "global", POINT),
            ("degree_selected", "per-k", POINT + ("degree",)),
        ):
            chosen = select_paper_rows(
                values, path, selection=selection, point_fields=fields
            )
            flags[name].update(id(row) for row in chosen)
    return [
        row | {name: id(row) in chosen for name, chosen in flags.items()}
        for row in rows
    ]


def curve_label(backend, point, fixed):
    label = LABELS[BACKENDS.index(backend)]
    if fixed and backend.startswith("bits"):
        settings = []
        if backend == "bits-sq":
            settings.append(f"degree={point['degree']}")
        settings.extend(
            (f"block={point['block_size']}", f"items={point['items_per_thread']}")
        )
        label += " [" + ", ".join(settings) + "]"
    return label


def size_tick(value):
    return (
        f"{value / 1e6:.2f}M"
        if value >= 1e6
        else f"{value / 1e3:.1f}K"
        if value >= 1e3
        else str(value)
    )


def write_outputs(rows, output_dir):
    """Write complete summaries and per-scenario latency/speedup/degree plots."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import utils

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with (output_dir / "application-scaling.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "operator":
            groups[row["operator"], row["mode"]].append(row)
    for (operator, mode), values in sorted(groups.items()):
        stem = f"{operator}-scaling" + ("-retention" if mode == "retention" else "")
        scenarios = sorted({row["scenario"] for row in values})
        for fixed in (False, True):
            field = "scaling_global_selected" if fixed else "scaling_selected"
            suffix = "-paper-global" if fixed else "-paper"
            with PdfPages(output_dir / f"{stem}{suffix}.pdf") as pdf:
                for scenario in scenarios:
                    points = [
                        row
                        for row in values
                        if row[field] and row["scenario"] == scenario
                    ]
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    handles, labels = [], []
                    for index, backend in enumerate(BACKENDS):
                        curve = sorted(
                            (row for row in points if row["backend"] == backend),
                            key=lambda row: row["size"],
                        )
                        if not curve:
                            continue
                        x = [row["size"] for row in curve]
                        latency = [row["median_ms"] for row in curve]
                        error = [
                            [max(0, row["median_ms"] - row["p25_ms"]) for row in curve],
                            [max(0, row["p75_ms"] - row["median_ms"]) for row in curve],
                        ]
                        style = dict(
                            color=utils.COLORS[index],
                            marker=utils.SHAPES[index],
                            capsize=3,
                            linewidth=1.5,
                        )
                        artist = axes[0].errorbar(x, latency, yerr=error, **style)
                        speedups, errors = speedup_errors(curve)
                        axes[1].errorbar(x, speedups, yerr=errors, **style)
                        handles.append(artist.lines[0])
                        labels.append(curve_label(backend, curve[0], fixed))
                    axes[0].set_yscale("log")
                    axes[0].set_ylabel("Full operator time [ms]")
                    axes[1].set_ylabel("Speedup vs AIR Top-K [×]")
                    axes[1].axhline(1, color="gray", linestyle=":", linewidth=0.8)
                    axes[1].set_ylim(bottom=0)
                    sizes = sorted({row["size"] for row in points})
                    detail = (
                        f"k = {scenario}"
                        if mode == "fixed-k"
                        else f"Target retention = {scenario * 100:g}%"
                    )
                    for ax in axes:
                        ax.set_xscale("log", base=2)
                        ax.set_xticks(sizes, labels=[size_tick(size) for size in sizes])
                        ax.set_xlabel(f"{SIZE_LABELS[operator]}\n{detail}")
                        ax.grid(alpha=0.35, linestyle="--")
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        frameon=False,
                        ncol=2 if fixed else 4,
                        fontsize=9,
                    )
                    fig.tight_layout(rect=(0, 0.12 if fixed else 0.08, 1, 1))
                    pdf.savefig(fig)
                    plt.close(fig)
        with PdfPages(output_dir / f"{stem}-degree.pdf") as pdf:
            for scenario in scenarios:
                fig, ax = plt.subplots(figsize=(8, 5))
                points = [
                    row
                    for row in values
                    if row["backend"] == "bits-sq"
                    and row["degree_selected"]
                    and row["scenario"] == scenario
                ]
                workload_ids = sorted(
                    {row["workload_id"] for row in points},
                    key=lambda key: next(
                        row["size"] for row in points if row["workload_id"] == key
                    ),
                )
                for i, workload in enumerate(workload_ids):
                    curve = sorted(
                        (row for row in points if row["workload_id"] == workload),
                        key=lambda row: row["degree"],
                    )
                    speeds, errors = speedup_errors(curve)
                    ax.errorbar(
                        [row["degree"] for row in curve],
                        speeds,
                        yerr=errors,
                        label=curve[0]["workload_label"],
                        color=utils.COLORS[i % len(utils.COLORS)],
                        marker=utils.SHAPES[i],
                        capsize=3,
                    )
                ax.set_xscale("log", base=2)
                degrees = sorted({row["degree"] for row in points})
                ax.set_xticks(degrees, labels=[str(value) for value in degrees])
                ax.set_xlabel("Split degree")
                ax.set_ylabel("Full operator speedup vs AIR Top-K [×]")
                ax.set_ylim(bottom=0)
                ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
                detail = (
                    f"k={scenario}"
                    if mode == "fixed-k"
                    else f"target retention={scenario * 100:g}%"
                )
                ax.set_title(
                    f"{operator}: {detail}\n"
                    "Best measured block/items at each degree; backend IQR"
                )
                ax.grid(alpha=0.35, linestyle="--")
                ax.legend(frameon=False, fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
