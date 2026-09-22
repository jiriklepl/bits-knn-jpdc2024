"""Shared configuration selection and rendering for application speedup plots."""

from collections import defaultdict
import csv
import math


def select_paper_rows(rows, path, *, selection="per-k"):
    """Choose bits configurations per k or across all k within one workload.

    Carry that configuration into every phase, including isolated selection.
    Only split bits may vary degree. Ties prefer block, items, then degree.
    """
    if selection not in ("per-k", "global"):
        raise ValueError(f"Unknown paper selection mode: {selection}")
    configurations = defaultdict(set)
    operators = defaultdict(list)
    seen = set()
    for row in rows:
        if row["backend"] == "block-select":
            continue
        config = tuple(
            row[name] for name in ("degree", "block_size", "items_per_thread")
        )
        point = (row["backend"], row["k"])
        key = point + (row["phase"],) + config
        if key in seen:
            raise ValueError(
                f"{path}: duplicate configuration for the same backend+k+phase point"
            )
        seen.add(key)
        if row["backend"] in ("bits", "bits-prefetch") and row["degree"] != 1:
            raise ValueError(f"{path}: ordinary bits requires degree=1")
        fixed = () if row["backend"] in ("bits", "bits-prefetch", "bits-sq") else config
        configurations[row["backend"]].add(fixed)
        if row["phase"] == "operator":
            operators[point].append(row)
    if any(len(values) > 1 for values in configurations.values()):
        raise ValueError(
            f"{path}: paper plots require one fixed configuration per backend "
            "apart from bits block size/items and split bits degree"
        )
    if selection == "per-k":
        winners = {
            point: min(
                values,
                key=lambda row: (
                    row["median_ms"],
                    row["block_size"],
                    row["items_per_thread"],
                    row["degree"],
                ),
            )
            for point, values in operators.items()
        }
    else:
        candidates = defaultdict(lambda: defaultdict(dict))
        for (backend, k), values in operators.items():
            for row in values:
                config = (row["block_size"], row["items_per_thread"], row["degree"])
                candidates[backend][config][k] = row
        winners = {}
        for backend, configs in candidates.items():
            ks = {k for values in configs.values() for k in values}
            complete = {
                config: values
                for config, values in configs.items()
                if set(values) == ks
            }
            if not complete:
                raise ValueError(
                    f"{path}: global paper selection requires a "
                    "block/item/degree configuration "
                    f"measured at every k for {backend}"
                )
            # All eligible configurations share the same k values and AIR baselines.
            # Minimizing mean log latency therefore maximizes geometric-mean
            # AIR speedup, with equal weight per k and no product overflow.
            config = min(
                complete,
                key=lambda config: (
                    math.fsum(
                        math.log(complete[config][k]["median_ms"]) for k in sorted(ks)
                    )
                    / len(ks),
                    config,
                ),
            )
            winners.update({(backend, k): row for k, row in complete[config].items()})
    selected = []
    for row in rows:
        if row["backend"] == "block-select":
            continue
        point = (row["backend"], row["k"])
        winner = winners.get(point)
        if winner is None:
            raise ValueError(f"{path}: paper selection requires full-operator timings")
        if all(
            row[name] == winner[name]
            for name in ("degree", "block_size", "items_per_thread")
        ):
            selected.append(row)
    if selection == "global":
        selected = select_global_bits_variant(selected, path)
    return selected


def select_global_bits_variant(rows, path):
    """Keep one bits curve per workload, chosen by full-operator geometric mean.

    The selected backend and configuration carry into every measured phase.
    Prefer ordinary (prefetched) bits on an exact tie.
    """
    ordinary = (
        "bits-prefetch"
        if any(r["backend"] == "bits-prefetch" for r in rows)
        else "bits"
    )
    backends = (ordinary, "bits-sq")
    operators = defaultdict(dict)
    for row in rows:
        if row["backend"] in backends and row["phase"] == "operator":
            operators[row["backend"]][row["k"]] = row
    if not operators:
        return rows
    ks = {k for points in operators.values() for k in points}
    complete = [backend for backend in backends if set(operators[backend]) == ks]
    if not complete:
        raise ValueError(f"{path}: global bits comparison requires every k")
    winner = min(
        complete,
        key=lambda backend: (
            math.fsum(math.log(operators[backend][k]["median_ms"]) for k in sorted(ks))
            / len(ks),
            backends.index(backend),
        ),
    )
    return [
        row
        for row in rows
        if row["backend"] not in ("bits", "bits-prefetch", "bits-sq")
        or row["backend"] == winner
    ]


def configuration_pages(pages, *, paper):
    """Bound detailed page size by showing one split degree with all comparisons.

    Repeat every ordinary bits variant and baseline for each split-degree page.
    Single-degree inputs retain their existing one-page layout.
    """
    for key, rows in sorted(pages.items()):
        degrees = sorted({r["degree"] for r in rows if r["backend"] == "bits-sq"})
        if paper or len(degrees) <= 1:
            yield key, None, rows
        else:
            for degree in degrees:
                yield (
                    key,
                    degree,
                    [
                        row
                        for row in rows
                        if row["backend"] != "bits-sq" or row["degree"] == degree
                    ],
                )


def write_configuration_csv(path, rows, *, selection):
    """Record every displayed point, its configuration, timings and provenance."""
    records = [row | {"selection_mode": selection} for row in rows]
    fields = list(dict.fromkeys(field for row in records for field in row))
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_combined_paper(rows, output_dir):
    """Write one three-application PDF and configuration CSV per size/mode/phase."""
    import matplotlib.pyplot as plt
    import utils

    applications = (
        ("database-topn", "Database top-N"),
        ("token-sampling", "Token sampling"),
        ("gradient-compression", "Gradient compression"),
    )
    backends = ("bits-prefetch", "bits-sq", "air-topk", "grid-select", "bits")
    labels = ("bits", "bits (split)", "AIR Top-K", "GridSelect", "bits (no prefetch)")
    # Select independently for each application/size, never across applications.
    cases = defaultdict(list)
    for row in rows:
        cases[row["size_tier"], row["operator"]].append(row)
    selections = {
        (tier, application, mode): select_paper_rows(
            values, values[0]["source_csv"], selection=mode
        )
        for (tier, application), values in cases.items()
        for mode in ("per-k", "global")
    }
    for tier in ("small", "middle", "large"):
        for mode in ("per-k", "global"):
            suffix = "-global" if mode == "global" else ""
            for phase, phase_name in (
                ("operator", "operator"),
                ("selection_isolated", "selection"),
            ):
                stem = f"applications-{tier}-paper{suffix}-{phase_name}"
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
                handles, records = {}, []
                for ax, (application, title) in zip(axes, applications):
                    values = [
                        row
                        for row in selections[tier, application, mode]
                        if row["phase"] == phase
                    ]
                    for index, backend in enumerate(backends):
                        points = sorted(
                            (row for row in values if row["backend"] == backend),
                            key=lambda row: row["k"],
                        )
                        if not points:
                            continue
                        handles[backend] = add_speedup_series(
                            ax,
                            points,
                            label=labels[index],
                            color=utils.COLORS[index],
                            marker=utils.SHAPES[index],
                            linestyle="-",
                        )
                        records.extend(row | {"label": labels[index]} for row in points)
                    ks = sorted({row["k"] for row in values})
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(ks, labels=[str(k) for k in ks])
                    ax.set_xlabel("k")
                    ax.set_title(title)
                    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
                    ax.grid(alpha=0.4, linestyle="--")
                axes[0].set_ylabel("Speedup vs AIR Top-K [×]")
                fit_speedup_axes(axes)
                ordered = [backend for backend in backends if backend in handles]
                fig.legend(
                    [handles[backend] for backend in ordered],
                    [labels[backends.index(backend)] for backend in ordered],
                    loc="lower center",
                    ncol=len(ordered),
                    frameon=False,
                )
                fig.tight_layout(rect=(0, 0.12, 1, 1))
                fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
                plt.close(fig)
                write_configuration_csv(
                    output_dir / f"{stem}-configs.csv", records, selection=mode
                )


def annotate_paper_selection(summary, workload, path):
    """Preserve all summary rows and mark the configurations used in the paper."""
    pages = defaultdict(list)
    for row in summary:
        pages[tuple(row[name] for name in workload)].append(row)
    selections = {
        field: {
            id(row)
            for rows in pages.values()
            for row in select_paper_rows(rows, path, selection=selection)
        }
        for field, selection in (
            ("paper_selected", "per-k"),
            ("paper_global_selected", "global"),
        )
    }
    return [
        row | {field: id(row) in chosen for field, chosen in selections.items()}
        for row in summary
    ]


def speedup_errors(points):
    """Invert backend quartiles around a fixed AIR median, not ratio uncertainty."""
    centers = [row["speedup_vs_air"] for row in points]
    lower = [
        max(0.0, value - value * (row["median_ms"] / row["p75_ms"]))
        for row, value in zip(points, centers)
    ]
    upper = [
        max(0.0, value * (row["median_ms"] / row["p25_ms"]) - value)
        for row, value in zip(points, centers)
    ]
    return centers, [lower, upper]


def add_speedup_series(ax, points, **style):
    """Draw a median/IQR curve and return its line/marker for a clean legend."""
    centers, errors = speedup_errors(points)
    artist = ax.errorbar(
        [row["k"] for row in points],
        centers,
        yerr=errors,
        capsize=3,
        linewidth=1.5,
        **style,
    )
    return artist.lines[0]


def fit_speedup_axes(axes):
    """Finalize limits after every panel, including error bars, has been drawn."""
    limits = [value for ax in axes for value in ax.dataLim.intervaly]
    finite = [value for value in limits if math.isfinite(value)]
    low, high = min(finite + [1.0]), max(finite + [1.0])
    padding = 0.05 * max(high - low, 1.0)
    # Include zero and leave room for low-valued markers and error-bar caps.
    bottom, top = min(0.0, low - padding), high + padding
    for ax in axes:
        ax.set_ylim(bottom, top)
