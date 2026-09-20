"""Shared error bars, legend handles and limits for application speedup plots."""

import math


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
