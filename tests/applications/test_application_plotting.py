"""Render regression checks; plotting packages are optional for native workflows."""

import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_plotting import speedup_errors  # noqa: E402


PLOTTING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "matplotlib", "seaborn")
)
BACKENDS = ("bits-prefetch", "bits-sq", "air-topk", "grid-select", "block-select")
PHASES = ("operator", "selection_isolated")


class SpeedupErrorTests(unittest.TestCase):
    def test_backend_quartiles_are_inverted_using_a_fixed_air_median(self):
        # AIR median = 8, backend median = 4: center 2, interval [1, 8].
        points = [
            dict(median_ms=4, p25_ms=1, p75_ms=8, speedup_vs_air=2),
            # Symmetric latency quartiles produce asymmetric speedup bars.
            dict(median_ms=4, p25_ms=2, p75_ms=6, speedup_vs_air=3),
            # These latency quartiles instead give symmetric speedup bars.
            dict(median_ms=4, p25_ms=8 / 3, p75_ms=8, speedup_vs_air=2),
            dict(median_ms=4, p25_ms=4, p75_ms=4, speedup_vs_air=2),
        ]
        centers, (lower, upper) = speedup_errors(points)
        self.assertEqual(centers, [2, 3, 2, 2])
        for actual, expected in zip(lower, [1, 1, 1, 0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(upper, [6, 3, 1, 0]):
            self.assertAlmostEqual(actual, expected)

    def test_tiny_positive_latencies_preserve_finite_speedup_errors(self):
        centers, errors = speedup_errors(
            [dict(median_ms=4e-300, p25_ms=1e-300, p75_ms=8e-300, speedup_vs_air=2)]
        )
        self.assertEqual(centers, [2])
        self.assertEqual(errors, [[1], [6]])


@unittest.skipUnless(
    PLOTTING_AVAILABLE, "Install plotting requirements to enable Agg render checks"
)
class ApplicationRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = tempfile.TemporaryDirectory(prefix="application-plot-cache-")
        cls.addClassCleanup(cache.cleanup)
        environment = patch.dict(os.environ, {"MPLCONFIGDIR": cache.name})
        environment.start()
        cls.addClassCleanup(environment.stop)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.container import ErrorbarContainer
        from matplotlib.lines import Line2D
        import tensor_analysis

        spec = importlib.util.spec_from_file_location(
            "database_render_analysis", SCRIPTS / "plot-database-topn.py"
        )
        cls.database = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.database)
        cls.tensor = tensor_analysis
        cls.plt, cls.np = plt, np
        cls.LineCollection = LineCollection
        cls.ErrorbarContainer = ErrorbarContainer
        cls.Line2D = Line2D

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(self.plt.close, "all")
        self.root = Path(temporary.name)

    def summary(self, operator, peak_phase):
        phases = ("operator",) if operator == "database-topn" else PHASES
        result = []
        for phase in phases:
            for backend in BACKENDS:
                for k in (32, 128):
                    speedup = {
                        "bits-prefetch": 0.5,
                        "bits-sq": 40 if phase == peak_phase else 2,
                        "air-topk": 1,
                        "grid-select": 1e-9,
                        "block-select": 0.2,
                    }[backend]
                    if backend == "bits-sq" and k == 128:
                        speedup *= 1.5
                    # Tiny positive durations keep the ratio meaningful while
                    # exercising small values. GridSelect lies near y = 0.
                    median = 1e-12 / speedup
                    quartiles = (0.002, 50) if backend == "bits-sq" else (0.8, 1.3)
                    row = dict(
                        dataset_id="a" * 64,
                        backend=backend,
                        rows=4096,
                        k=k,
                        retention_ratio=k / 4096,
                        degree=32 if backend == "bits-sq" else 1,
                        block_size=0 if backend == "grid-select" else 512,
                        items_per_thread=4,
                        phase=phase,
                        samples=30,
                        median_ms=median,
                        p25_ms=median * quartiles[0],
                        p75_ms=median * quartiles[1],
                        speedup_vs_air=speedup,
                    )
                    if operator != "database-topn":
                        row.update(
                            operator=operator,
                            batch_size=8 if operator == "token-sampling" else 1,
                            temperature=0.8 if operator == "token-sampling" else 0,
                            seed=42 if operator == "token-sampling" else 0,
                        )
                    result.append(row)
        return result

    def render(self, operator, summary):
        captured = []

        class CapturePdf:
            def __init__(self, filename):
                self.filename = Path(filename)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def savefig(self, figure):
                # Execute the actual Agg renderer and layout, not mocked axes.
                figure.canvas.draw()
                captured.append((self.filename, figure))

        path = self.root / f"{operator}-test-1.csv"
        with patch("matplotlib.backends.backend_pdf.PdfPages", CapturePdf):
            if operator == "database-topn":
                for paper in (False, True):
                    self.database.validate_plot_summary(summary, path, paper)
                    self.database.plot(summary, path, self.root, paper=paper)
            else:
                self.tensor.plot_pages(summary, path)
                self.tensor.plot(summary, path, self.root)
        self.assertEqual(len(captured), 2)
        self.assertEqual(
            {filename.name for filename, _ in captured},
            {f"{path.stem}.pdf", f"{path.stem}-paper.pdf"},
        )
        return captured

    def assert_render(self, operator, summary, filename, figure):
        paper = filename.stem.endswith("-paper")
        phases = ("operator",) if operator == "database-topn" else PHASES
        visible = BACKENDS[:-1] if paper else BACKENDS
        self.assertEqual(len(figure.axes), len(phases))
        first_lines = []
        for axis, phase in zip(figure.axes, phases):
            containers = [
                item
                for item in axis.containers
                if isinstance(item, self.ErrorbarContainer)
            ]
            self.assertEqual(len(containers), len(visible))
            endpoints = []
            for container, backend in zip(containers, visible):
                points = [
                    row
                    for row in summary
                    if row["phase"] == phase and row["backend"] == backend
                ]
                points.sort(key=lambda row: row["k"])
                line, caps, collections = container.lines
                self.assertIsInstance(line, self.Line2D)
                self.assertTrue(container.has_yerr)
                self.assertTrue(caps)
                self.assertTrue(collections)
                self.assertTrue(
                    all(isinstance(item, self.LineCollection) for item in collections)
                )
                self.np.testing.assert_allclose(
                    line.get_ydata(orig=False),
                    [row["speedup_vs_air"] for row in points],
                )
                intervals = [
                    segment[:, 1]
                    for item in collections
                    for segment in item.get_segments()
                ]
                self.assertEqual(len(intervals), len(points))
                for interval, row in zip(intervals, points):
                    air = row["median_ms"] * row["speedup_vs_air"]
                    self.np.testing.assert_allclose(
                        interval, [air / row["p75_ms"], air / row["p25_ms"]]
                    )
                    endpoints.extend(interval)
                endpoints.extend(line.get_ydata())
                if phase == "operator":
                    first_lines.append(line)
            self.assertTrue(all(math.isfinite(value) for value in endpoints))
            lower, upper = axis.get_ylim()
            self.assertTrue(math.isfinite(lower) and math.isfinite(upper))
            # Padding protects marker/cap glyphs, including points near zero.
            clearance = 0.02 * (upper - lower)
            self.assertLess(lower + clearance, min(endpoints))
            self.assertGreater(upper - clearance, max(endpoints))
            if paper:
                self.assertEqual(axis.get_title(), "")
                self.assertIn(
                    "operator" if phase == "operator" else "selection",
                    axis.get_ylabel().lower(),
                )
            else:
                self.assertTrue(axis.get_title())
        if len(figure.axes) == 2:
            self.assertEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())

        legends = list(figure.legends) + [
            axis.get_legend() for axis in figure.axes if axis.get_legend() is not None
        ]
        self.assertEqual(len(legends), 1)
        legend = legends[0]
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        self.assertEqual(len(handles), len(visible))
        self.assertTrue(all(isinstance(handle, self.Line2D) for handle in handles))
        self.assertEqual(
            [handle.get_marker() for handle in handles],
            [line.get_marker() for line in first_lines],
        )
        self.assertIn("s", [handle.get_marker() for handle in handles])
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(any("BlockSelect" in label for label in labels), not paper)
        for label in labels:
            self.assertEqual("degree=" in label, not paper)
        title = figure._suptitle.get_text() if figure._suptitle is not None else ""
        if paper:
            self.assertEqual(title, "")
        elif len(figure.axes) == 2:
            self.assertTrue(title)

    def check_operator(self, operator):
        for peak in ("operator",) if operator == "database-topn" else PHASES:
            with self.subTest(operator=operator, largest_panel=peak):
                summary = self.summary(operator, peak)
                if operator != "database-topn":
                    maxima = {
                        phase: max(
                            row["speedup_vs_air"]
                            for row in summary
                            if row["phase"] == phase
                        )
                        for phase in PHASES
                    }
                    self.assertGreater(
                        maxima[peak],
                        maxima[PHASES[1] if peak == PHASES[0] else PHASES[0]],
                    )
                for filename, figure in self.render(operator, summary):
                    with self.subTest(paper=filename.stem.endswith("-paper")):
                        self.assert_render(operator, summary, filename, figure)

    def test_database_render_has_unclipped_errors_and_clean_legend(self):
        self.check_operator("database-topn")

    def test_sampling_shared_limits_include_both_panels_and_errors(self):
        self.check_operator("token-sampling")

    def test_gradient_shared_limits_include_both_panels_and_errors(self):
        self.check_operator("gradient-compression")


if __name__ == "__main__":
    unittest.main()
