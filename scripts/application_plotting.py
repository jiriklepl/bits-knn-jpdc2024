"""Shared configuration selection and rendering for application speedup plots."""

from collections import defaultdict
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
    return selected


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


def global_configuration_label(backend, row):
    """Describe the one configuration shared by all points of a global curve."""
    if backend not in ("bits", "bits-prefetch", "bits-sq"):
        return ""
    degree = f"degree={row['degree']}, " if backend == "bits-sq" else ""
    return f" [{degree}block={row['block_size']}, items={row['items_per_thread']}]"


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
