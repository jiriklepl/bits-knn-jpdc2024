"""Render regression checks; plotting packages are optional for native workflows."""

import importlib.util
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_plotting import (  # noqa: E402
    annotate_paper_selection,
    plot_combined_paper,
    select_paper_rows,
    speedup_errors,
)


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


class PaperSelectionTests(unittest.TestCase):
    def rows(self, block, operator_ms, selection_ms, **settings):
        config = (
            dict(
                dataset_id="a",
                backend="bits-sq",
                k=32,
                degree=32,
                block_size=block,
                items_per_thread=4,
            )
            | settings
        )
        if "degree" not in settings and config["backend"] in ("bits", "bits-prefetch"):
            config["degree"] = 1
        return [
            config | dict(phase="operator", median_ms=operator_ms),
            config | dict(phase="selection_isolated", median_ms=selection_ms),
        ]

    def test_operator_winner_carries_its_own_selection_measurements(self):
        rows = (
            self.rows(128, 3, 0.1)
            + self.rows(256, 2, 0.5, items_per_thread=13)
            + self.rows(256, 3, 0.01, items_per_thread=7)
            + self.rows(512, 4, 0.2)
        )
        chosen = select_paper_rows(rows, "test.csv")
        self.assertEqual([row["block_size"] for row in chosen], [256, 256])
        self.assertEqual([row["items_per_thread"] for row in chosen], [13, 13])
        self.assertEqual([row["median_ms"] for row in chosen], [2, 0.5])

    def test_exact_ties_use_smaller_blocks_then_fewer_items_then_degree(self):
        rows = (
            self.rows(512, 2, 0.1, items_per_thread=4)
            + self.rows(128, 2, 0.5, items_per_thread=13)
            + self.rows(128, 2, 0.5, items_per_thread=7)
            + self.rows(128, 2, 0.5, items_per_thread=7, degree=8)
        )
        for selection in ("per-k", "global"):
            for values in (rows, list(reversed(rows))):
                self.assertEqual(
                    {
                        (r["block_size"], r["items_per_thread"], r["degree"])
                        for r in select_paper_rows(
                            values, "test.csv", selection=selection
                        )
                    },
                    {(128, 7, 8)},
                )

    def test_annotations_keep_workloads_k_and_backend_ids_separate(self):
        rows = []
        expected = {}
        for dataset in ("a", "b"):
            for k in (32, 64):
                for backend in ("bits", "bits-prefetch", "bits-sq"):
                    winner = 128 if (dataset == "a") == (k == 32) else 512
                    if backend == "bits":
                        winner = 512 if winner == 128 else 128
                    expected[dataset, k, backend] = winner
                    for block in (128, 512):
                        rows += self.rows(
                            block,
                            1 if block == winner else 2,
                            0.5,
                            dataset_id=dataset,
                            k=k,
                            backend=backend,
                        )
        rows += self.rows(128, 0.01, 0.01, backend="block-select")
        annotated = annotate_paper_selection(rows, ("dataset_id",), "test.csv")
        self.assertEqual(len(annotated), len(rows))
        self.assertTrue(all("paper_selected" not in row for row in rows))
        for row in annotated:
            chosen = (
                row["backend"] != "block-select"
                and row["block_size"]
                == expected[row["dataset_id"], row["k"], row["backend"]]
            )
            self.assertEqual(row["paper_selected"], chosen)

    def test_duplicate_points_and_missing_operator_are_rejected(self):
        rows = self.rows(128, 2, 1)
        with self.assertRaisesRegex(ValueError, "duplicate configuration"):
            select_paper_rows(rows + rows, "test.csv")
        with self.assertRaisesRegex(ValueError, "full-operator timings"):
            select_paper_rows(rows[1:], "test.csv")

    def test_global_uses_geometric_mean_instead_of_total_time_or_per_k_winners(self):
        # 128 has geometric mean 10; 256 has sqrt(160), but a smaller total.
        rows = (
            self.rows(128, 1, 0.9, k=32)
            + self.rows(128, 100, 90, k=64)
            + self.rows(256, 4, 0.1, k=32)
            + self.rows(256, 40, 0.1, k=64)
        )
        per_k = select_paper_rows(rows, "test.csv")
        global_rows = select_paper_rows(rows, "test.csv", selection="global")
        self.assertEqual(
            {(r["k"], r["block_size"]) for r in per_k}, {(32, 128), (64, 256)}
        )
        self.assertEqual({r["block_size"] for r in global_rows}, {128})
        self.assertEqual(
            {r["median_ms"] for r in global_rows if r["phase"] == "selection_isolated"},
            {0.9, 90},
        )

    def test_global_candidates_must_cover_every_k(self):
        partial = self.rows(128, 0.01, 0.01, k=32)
        complete = self.rows(256, 2, 1, k=32) + self.rows(256, 2, 1, k=64)
        chosen = select_paper_rows(partial + complete, "test.csv", selection="global")
        self.assertEqual({row["block_size"] for row in chosen}, {256})
        with self.assertRaisesRegex(ValueError, "measured at every k"):
            select_paper_rows(partial + complete[2:], "test.csv", selection="global")

    def test_global_keeps_items_fixed_and_does_not_combine_partial_pairs(self):
        # The same block's different item counts are separate candidates.
        partial = self.rows(128, 0.01, 0.01, k=32, items_per_thread=7) + self.rows(
            128, 0.01, 0.01, k=64, items_per_thread=13
        )
        complete = self.rows(128, 2, 1, k=32, items_per_thread=8) + self.rows(
            128, 2, 1, k=64, items_per_thread=8
        )
        chosen = select_paper_rows(partial + complete, "test.csv", selection="global")
        self.assertEqual({r["items_per_thread"] for r in chosen}, {8})
        with self.assertRaisesRegex(ValueError, "measured at every k"):
            select_paper_rows(partial, "test.csv", selection="global")

    def test_degree_sweep_selects_operator_winner_and_marks_each_phase(self):
        rows = (
            self.rows(128, 1, 9, k=32, degree=8)
            + self.rows(128, 9, 1, k=64, degree=8)
            + self.rows(128, 4, 0.1, k=32, degree=32)
            + self.rows(128, 4, 0.1, k=64, degree=32)
        )
        annotated = annotate_paper_selection(rows, ("dataset_id",), "test.csv")
        for row in annotated:
            expected = 8 if row["k"] == 32 else 32
            self.assertEqual(row["paper_selected"], row["degree"] == expected)
            self.assertEqual(row["paper_global_selected"], row["degree"] == 8)

    def test_global_never_combines_incomplete_degrees_at_one_block_item_pair(self):
        partial = self.rows(128, 0.1, 1, k=32, degree=8) + self.rows(
            128, 0.1, 1, k=64, degree=32
        )
        complete = self.rows(128, 2, 1, k=32, degree=128) + self.rows(
            128, 2, 1, k=64, degree=128
        )
        chosen = select_paper_rows(partial + complete, "test.csv", selection="global")
        self.assertEqual({row["degree"] for row in chosen}, {128})
        with self.assertRaisesRegex(ValueError, "measured at every k"):
            select_paper_rows(partial, "test.csv", selection="global")

    def test_ordinary_bits_requires_degree_one_and_baselines_remain_fixed(self):
        for backend in ("bits", "bits-prefetch"):
            for selection in ("per-k", "global"):
                with self.subTest(backend=backend, selection=selection):
                    with self.assertRaisesRegex(ValueError, "degree=1"):
                        select_paper_rows(
                            self.rows(128, 1, 1, backend=backend, degree=8),
                            "test.csv",
                            selection=selection,
                        )
        for backend in ("air-topk", "grid-select"):
            for dimension in ("degree", "block_size", "items_per_thread"):
                rows = self.rows(128, 1, 1, backend=backend)
                rows += [row | {dimension: row[dimension] * 2, "k": 64} for row in rows]
                with self.subTest(backend=backend, dimension=dimension):
                    with self.assertRaisesRegex(ValueError, "fixed configuration"):
                        select_paper_rows(rows, "test.csv")

    def test_global_ties_and_large_values_are_deterministic(self):
        rows = (
            self.rows(512, 2e200, 1, k=32)
            + self.rows(512, 8e200, 1, k=64)
            + self.rows(128, 8e200, 1, k=32)
            + self.rows(128, 2e200, 1, k=64)
        )
        for values in (rows, list(reversed(rows))):
            chosen = select_paper_rows(values, "test.csv", selection="global")
            self.assertEqual({row["block_size"] for row in chosen}, {128})

    def test_global_annotations_keep_workloads_and_backends_separate(self):
        rows = []
        for dataset in ("a", "b"):
            for backend in ("bits-prefetch", "bits-sq"):
                winner = (
                    128 if (dataset == "a") == (backend == "bits-prefetch") else 512
                )
                for block in (128, 512):
                    for k in (32, 64):
                        rows += self.rows(
                            block,
                            (0.5 if dataset == "b" and backend == "bits-sq" else 1)
                            if block == winner
                            else 2,
                            1,
                            dataset_id=dataset,
                            backend=backend,
                            k=k,
                        )
        annotated = annotate_paper_selection(rows, ("dataset_id",), "test.csv")
        for row in annotated:
            winner = (
                128
                if (row["dataset_id"] == "a") == (row["backend"] == "bits-prefetch")
                else 512
            )
            backend = "bits-prefetch" if row["dataset_id"] == "a" else "bits-sq"
            self.assertEqual(
                row["paper_global_selected"],
                row["block_size"] == winner and row["backend"] == backend,
            )
        self.assertTrue(all("paper_global_selected" not in row for row in rows))

    def test_global_variant_uses_operator_geometric_mean_and_carries_selection(self):
        rows = (
            self.rows(128, 1, 99, k=32, backend="bits-prefetch")
            + self.rows(128, 100, 99, k=64, backend="bits-prefetch")
            + self.rows(256, 4, 0.1, k=32)
            + self.rows(256, 40, 0.1, k=64)
        )
        chosen = select_paper_rows(rows, "test.csv", selection="global")
        self.assertEqual({r["backend"] for r in chosen}, {"bits-prefetch"})
        self.assertEqual(
            [r["median_ms"] for r in chosen if r["phase"] == "selection_isolated"],
            [99, 99],
        )
        self.assertEqual(
            {r["backend"] for r in select_paper_rows(rows, "test.csv")},
            {"bits-prefetch", "bits-sq"},
        )


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

            def savefig(self, figure, **kwargs):
                # Execute the actual Agg renderer and layout, not mocked axes.
                figure.canvas.draw()
                captured.append((self.filename, figure))

        path = self.root / f"{operator}-test-1.csv"
        with patch("matplotlib.backends.backend_pdf.PdfPages", CapturePdf):
            if operator == "database-topn":
                for paper, selection in (
                    (False, "per-k"),
                    (True, "per-k"),
                    (True, "global"),
                ):
                    self.database.plot(
                        summary, path, self.root, paper=paper, selection=selection
                    )
            else:
                self.tensor.plot_pages(summary, path)
                self.tensor.plot(summary, path, self.root)
        degrees = {row["degree"] for row in summary if row["backend"] == "bits-sq"}
        self.assertEqual(len(captured), max(1, len(degrees)) + 2)
        self.assertEqual(
            {filename.name for filename, _ in captured},
            {
                f"{path.stem}.pdf",
                f"{path.stem}-paper.pdf",
                f"{path.stem}-paper-global.pdf",
            },
        )
        return captured

    def test_combined_papers_keep_application_winners_independent_and_export_points(
        self
    ):
        rows = []
        applications = ("database-topn", "token-sampling", "gradient-compression")
        for tier in ("small", "middle", "large"):
            for application in applications:
                for row in self.summary("token-sampling", "operator"):
                    winner = (
                        "bits-sq"
                        if application == "token-sampling"
                        else "bits-prefetch"
                    )
                    speedup = 8 if row["backend"] == winner else 2
                    # Selection alone prefers the opposite bits variant.
                    if row["phase"] == "selection_isolated":
                        speedup = 1 / speedup
                    if row["backend"] == "air-topk":
                        speedup = 1
                    latency = 1 / speedup
                    rows.append(
                        row
                        | dict(
                            operator=application,
                            size_tier=tier,
                            source_csv=f"{application}-{tier}.csv",
                            median_ms=latency,
                            p25_ms=0.8 * latency,
                            p75_ms=1.2 * latency,
                            speedup_vs_air=speedup,
                        )
                    )
        captured = []

        def capture(figure, path, **kwargs):
            figure.canvas.draw()
            captured.append((Path(path), figure))

        with patch("matplotlib.figure.Figure.savefig", capture):
            plot_combined_paper(rows, self.root)
        self.assertEqual(len(captured), 12)
        for path, figure in captured:
            global_choice = "-global-" in path.name
            phase = "selection_isolated" if "-selection" in path.name else "operator"
            self.assertEqual(len(figure.axes), 3)
            with path.with_name(path.stem + "-configs.csv").open() as source:
                records = list(csv.DictReader(source))
            self.assertEqual({r["phase"] for r in records}, {phase})
            self.assertEqual(len(records), 18 if global_choice else 24)
            for ax, application in zip(figure.axes, applications):
                self.assertNotIn("operator", ax.get_ylabel().lower())
                self.assertNotIn("selection", ax.get_ylabel().lower())
                self.assertTrue(ax.get_title())
                points = [r for r in records if r["operator"] == application]
                if global_choice:
                    self.assertEqual(
                        {
                            r["backend"]
                            for r in points
                            if r["backend"].startswith("bits")
                        },
                        {
                            "bits-sq"
                            if application == "token-sampling"
                            else "bits-prefetch"
                        },
                    )
                for container in ax.containers:
                    label = container.get_label()
                    expected = [r for r in points if r["label"] == label]
                    self.np.testing.assert_allclose(
                        container.lines[0].get_ydata(orig=False),
                        [float(r["speedup_vs_air"]) for r in expected],
                    )
                    self.assertNotIn("=", label)

    def assert_render(self, operator, summary, filename, figure):
        paper = filename.stem.endswith(("-paper", "-paper-global"))
        phases = ("operator",) if operator == "database-topn" else PHASES
        visible = BACKENDS[:-1] if paper else BACKENDS
        if filename.stem.endswith("-paper-global"):
            visible = ("bits-sq", "air-topk", "grid-select")
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
        for label, backend in zip(labels, visible):
            self.assertEqual(
                "degree=" in label,
                not paper,
            )
        if paper:
            self.assertTrue(all("block=" not in label for label in labels))
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
                    with self.subTest(plot=filename.stem):
                        self.assert_render(operator, summary, filename, figure)

    def test_database_render_has_unclipped_errors_and_clean_legend(self):
        self.check_operator("database-topn")

    def test_sampling_shared_limits_include_both_panels_and_errors(self):
        self.check_operator("token-sampling")

    def test_gradient_shared_limits_include_both_panels_and_errors(self):
        self.check_operator("gradient-compression")

    def test_joint_sweeps_render_all_variants_and_only_operator_winners(self):
        for operator in ("database-topn", "token-sampling", "gradient-compression"):
            with self.subTest(operator=operator):
                summary = self.summary(operator, "operator")
                variants = []
                for row in summary:
                    if row["backend"] not in ("bits-prefetch", "bits-sq"):
                        variants.append(row)
                        continue
                    for block in (128, 256, 512):
                        for items in (4, 7, 8, 13, 16):
                            degrees = (
                                (8, 32, 128, 512)
                                if row["backend"] == "bits-sq"
                                else (1,)
                            )
                            for degree in degrees:
                                best_degree = (
                                    (8 if row["k"] == 32 else 32)
                                    if row["backend"] == "bits-sq"
                                    else 1
                                )
                                winner = (
                                    (128, 13, best_degree)
                                    if row["k"] == 32
                                    else (512, 7, best_degree)
                                )
                                # Reverse the isolated-selection ranking to ensure the
                                # paper's second panel carries the operator winner.
                                factor = 0.5 if (block, items, degree) == winner else 2
                                if (block, items, degree) == winner and row["k"] == 32:
                                    factor = 0.25  # Unique global winner at k=32.
                                if row["phase"] == "selection_isolated":
                                    factor = 1 / factor
                                variants.append(
                                    row
                                    | {
                                        "degree": degree,
                                        "block_size": block,
                                        "items_per_thread": items,
                                        "median_ms": row["median_ms"] * factor,
                                        "p25_ms": row["p25_ms"] * factor,
                                        "p75_ms": row["p75_ms"] * factor,
                                        "speedup_vs_air": row["speedup_vs_air"]
                                        / factor,
                                    }
                                )
                plotted_split_degrees = []
                for filename, figure in self.render(operator, variants):
                    paper = filename.stem.endswith(("-paper", "-paper-global"))
                    global_choice = filename.stem.endswith("-paper-global")
                    for axis, phase in zip(
                        figure.axes,
                        ("operator",) if operator == "database-topn" else PHASES,
                    ):
                        containers = [
                            c
                            for c in axis.containers
                            if isinstance(c, self.ErrorbarContainer)
                        ]
                        visible = (
                            ("bits-sq", "air-topk", "grid-select")
                            if global_choice
                            else BACKENDS[:-1]
                        )
                        self.assertEqual(len(containers), len(visible) if paper else 33)
                        if not paper:
                            if operator != "database-topn":
                                self.assertGreater(axis.bbox.height / figure.dpi, 2.5)
                            styles = {
                                (
                                    c.lines[0].get_color(),
                                    c.lines[0].get_marker(),
                                    c.lines[0].get_linestyle(),
                                )
                                for c in containers
                            }
                            self.assertEqual(len(styles), len(containers))
                        if paper:
                            for container, backend in zip(containers, visible):
                                expected = [
                                    row
                                    for row in variants
                                    if row["phase"] == phase
                                    and row["backend"] == backend
                                    and (
                                        backend not in ("bits-prefetch", "bits-sq")
                                        or (
                                            row["block_size"],
                                            row["items_per_thread"],
                                            row["degree"],
                                        )
                                        == (
                                            (128, 13, 8 if backend == "bits-sq" else 1)
                                            if global_choice or row["k"] == 32
                                            else (
                                                512,
                                                7,
                                                32 if backend == "bits-sq" else 1,
                                            )
                                        )
                                    )
                                ]
                                self.np.testing.assert_allclose(
                                    container.lines[0].get_ydata(orig=False),
                                    [row["speedup_vs_air"] for row in expected],
                                )
                    legend = (
                        figure.legends[0]
                        if figure.legends
                        else figure.axes[0].get_legend()
                    )
                    labels = [text.get_text() for text in legend.get_texts()]
                    if global_choice:
                        self.assertEqual(
                            labels,
                            ["bits (split)", "AIR Top-K", "GridSelect"],
                        )
                    if not paper:
                        split_labels = [
                            label
                            for label in labels
                            if label.startswith("bits (split)")
                        ]
                        degree = int(split_labels[0].split("degree=")[1].split(",")[0])
                        plotted_split_degrees.append(degree)
                        self.assertEqual(len(split_labels), 15)
                        self.assertTrue(
                            all(f"degree={degree}," in label for label in split_labels)
                        )
                        title = (
                            figure._suptitle.get_text()
                            if figure._suptitle
                            else figure.axes[0].get_title()
                        )
                        self.assertIn(f"split degree={degree}", title)
                        for block in (128, 256, 512):
                            self.assertEqual(
                                sum(
                                    f"block={block}," in label
                                    and label.startswith("bits")
                                    for label in labels
                                ),
                                10,
                            )
                    # All 33 detailed labels fit below the chart without overlap.
                    bounds = legend.get_window_extent(figure.canvas.get_renderer())
                    self.assertGreaterEqual(bounds.y0, figure.bbox.y0)
                    self.assertLessEqual(
                        bounds.y1, min(ax.bbox.y0 for ax in figure.axes)
                    )
                    self.assertGreaterEqual(bounds.x0, figure.bbox.x0)
                    self.assertLessEqual(bounds.x1, figure.bbox.x1)
                self.assertEqual(plotted_split_degrees, [8, 32, 128, 512])


if __name__ == "__main__":
    unittest.main()
