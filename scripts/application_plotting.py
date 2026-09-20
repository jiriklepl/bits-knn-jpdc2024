"""Shared configuration selection and rendering for application speedup plots."""

from collections import defaultdict
import math


def select_paper_rows(rows, path, *, selection="per-k"):
    """Choose BITS blocks per k or across all k within one workload.

    Carry that configuration into every phase, including isolated selection.
    Keep degree/items fixed so a block-size sweep has a single interpretation.
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
        fixed = (
            (row["degree"], row["items_per_thread"])
            if row["backend"] in ("bits", "bits-prefetch", "bits-sq")
            else config
        )
        configurations[row["backend"]].add(fixed)
        if row["phase"] == "operator":
            operators[point].append(row)
    if any(len(values) > 1 for values in configurations.values()):
        raise ValueError(
            f"{path}: paper plots require one fixed configuration per backend "
            "apart from BITS block size; keep degree and items per thread fixed"
        )
    if selection == "per-k":
        winners = {
            point: min(values, key=lambda row: (row["median_ms"], row["block_size"]))
            for point, values in operators.items()
        }
    else:
        candidates = defaultdict(lambda: defaultdict(dict))
        for (backend, k), values in operators.items():
            for row in values:
                candidates[backend][row["block_size"]][k] = row
        winners = {}
        for backend, blocks in candidates.items():
            ks = {k for values in blocks.values() for k in values}
            complete = {
                block: values for block, values in blocks.items() if set(values) == ks
            }
            if not complete:
                raise ValueError(
                    f"{path}: global paper selection requires a block size "
                    f"measured at every k for {backend}"
                )
            # All eligible blocks share the same k values and AIR baselines.
            # Minimizing mean log latency therefore maximizes geometric-mean
            # AIR speedup, with equal weight per k and no product overflow.
            block = min(
                complete,
                key=lambda block: (
                    math.fsum(
                        math.log(complete[block][k]["median_ms"]) for k in sorted(ks)
                    )
                    / len(ks),
                    block,
                ),
            )
            winners.update({(backend, k): row for k, row in complete[block].items()})
    selected = []
    for row in rows:
        if row["backend"] == "block-select":
            continue
        winner = winners.get((row["backend"], row["k"]))
        if winner is None:
            raise ValueError(f"{path}: paper selection requires full-operator timings")
        if all(
            row[name] == winner[name]
            for name in ("degree", "block_size", "items_per_thread")
        ):
            selected.append(row)
    return selected


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
