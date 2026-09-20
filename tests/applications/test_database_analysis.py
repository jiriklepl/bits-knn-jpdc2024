"""Check database timing aggregation without requiring plotting packages or a GPU."""

import copy
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot-database-topn.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("database_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "database-topn-host-1.csv"
        self.rows = self.measurements()

    def measurements(
        self,
        digest="a" * 64,
        degree=32,
        multiplier=1,
        backend="bits-sq",
        k=32,
        phase_factors=None,
        **settings,
    ):
        config = (
            dict(
                dataset_id=digest,
                backend=backend,
                rows=4096,
                k=k,
                retention_ratio=k / 4096,
                degree=degree,
                block_size=512,
                items_per_thread=4,
            )
            | settings
        )
        rows = [config | dict(iteration=-1, phase="upload_shared", seconds=0.1)]
        for iteration, seconds in enumerate([0.001, 0.003, 0.009]):
            for phase in sorted(analysis.MEASURED_PHASES):
                rows.append(
                    config
                    | dict(
                        iteration=iteration,
                        phase=phase,
                        seconds=seconds
                        * multiplier
                        * (phase_factors or {}).get(phase, 1),
                    )
                )
        return rows

    def summarize(self, rows):
        with self.path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return analysis.summarize(self.path)

    def test_keeps_all_measured_iterations_and_separate_phases(self):
        summary = self.summarize(self.rows)
        self.assertEqual(len(summary), 6)
        for row in summary:
            if row["phase"] == "upload_shared":
                self.assertEqual(row["samples"], 1)
                self.assertEqual(row["median_ms"], 100)
            else:
                self.assertEqual(row["samples"], 3)
                self.assertEqual(row["median_ms"], 3)
                self.assertEqual((row["p25_ms"], row["p75_ms"]), (2, 6))

    def test_does_not_pool_inputs_or_choose_best_configuration(self):
        rows = self.rows + self.measurements(degree=16, multiplier=2)
        rows += self.measurements(digest="b" * 64, multiplier=3)
        summary = [row for row in self.summarize(rows) if row["phase"] == "operator"]
        self.assertEqual(len(summary), 3)
        for actual, expected in zip(
            sorted(row["median_ms"] for row in summary), [3, 6, 9]
        ):
            self.assertAlmostEqual(actual, expected)

    def comparison(self):
        rows = []
        for backend in analysis.BACKENDS:
            rows.extend(self.measurements(backend=backend))
        return rows

    def test_retains_all_backends_and_uses_matching_air_phase(self):
        rows = self.measurements()
        rows += self.measurements(
            backend="air-topk", multiplier=2, phase_factors={"selection_isolated": 0.25}
        )
        rows += self.measurements(backend="block-select", multiplier=4)
        rows += self.measurements(backend="bits")
        rows += self.measurements(backend="bits-prefetch")
        rows += self.measurements(backend="grid-select")
        summary = self.summarize(rows)
        self.assertEqual({row["backend"] for row in summary}, set(analysis.BACKENDS))
        lookup = {(row["backend"], row["phase"]): row for row in summary}
        self.assertEqual(lookup["bits-sq", "operator"]["speedup_vs_air"], 2)
        self.assertEqual(lookup["bits-sq", "selection_isolated"]["speedup_vs_air"], 0.5)
        self.assertEqual(lookup["block-select", "operator"]["speedup_vs_air"], 0.5)

    def test_missing_or_ambiguous_air_baseline_prevents_plotting(self):
        for other in [
            [],
            self.measurements(backend="air-topk", digest="b" * 64),
            self.measurements(backend="air-topk", k=64),
            self.measurements(backend="air-topk", block_size=256)
            + self.measurements(backend="air-topk", block_size=512),
        ]:
            summary = self.summarize(self.rows + other)
            for paper in (False, True):
                with self.subTest(other=other, paper=paper), self.assertRaisesRegex(
                    ValueError, "unique matching AIR"
                ):
                    analysis.validate_plot_summary(summary, self.path, paper)

    def test_paper_rejects_duplicate_backend_k_and_disjoint_configuration_sweeps(self):
        for extra in [
            self.measurements(degree=16),
            self.measurements(degree=16, k=64)
            + self.measurements(backend="air-topk", k=64),
        ]:
            summary = self.summarize(self.comparison() + extra)
            analysis.validate_plot_summary(summary, self.path)
            with self.assertRaisesRegex(ValueError, "paper"):
                analysis.validate_plot_summary(summary, self.path, paper=True)

    def test_paper_render_has_short_names_no_titles_and_no_block_select(self):
        summary = self.summarize(self.comparison())
        plt = types.ModuleType("matplotlib.pyplot")
        figure, axes = MagicMock(), MagicMock()
        handles = []

        def errorbar(*args, **kwargs):
            handle = MagicMock(name="data-line")
            handles.append(handle)
            return types.SimpleNamespace(lines=(handle, (), ()))

        axes.errorbar.side_effect = errorbar
        plt.subplots = MagicMock(return_value=(figure, axes))
        plt.close = MagicMock()
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = plt
        backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")
        backend_pdf.PdfPages = MagicMock()
        utils = types.ModuleType("utils")
        utils.COLORS = ["blue"] * len(analysis.BACKENDS)
        utils.SHAPES = ["o"] * len(analysis.BACKENDS)
        modules = {
            "matplotlib": matplotlib,
            "matplotlib.pyplot": plt,
            "matplotlib.backends.backend_pdf": backend_pdf,
            "utils": utils,
        }
        with patch.dict(sys.modules, modules), patch.object(
            analysis, "fit_speedup_axes"
        ) as fit_axes:
            analysis.plot(summary, self.path, self.path.parent, paper=True)
        fit_axes.assert_called_once_with([axes])
        labels = [call.kwargs["label"] for call in axes.errorbar.call_args_list]
        self.assertEqual(labels, list(analysis.LABELS[:-1]))
        self.assertEqual(axes.legend.call_args.args, (handles, labels))
        axes.set_ylim.assert_not_called()
        axes.set_title.assert_not_called()
        figure.suptitle.assert_not_called()
        self.assertIn("speedup vs AIR", axes.set_ylabel.call_args.args[0])
        backend_pdf.PdfPages.assert_called_once_with(
            self.path.parent / f"{self.path.stem}-paper.pdf"
        )

        figure.reset_mock()
        axes.reset_mock()
        handles.clear()
        with patch.dict(sys.modules, modules), patch.object(
            analysis, "fit_speedup_axes"
        ) as fit_axes:
            analysis.plot(summary, self.path, self.path.parent, paper=False)
        fit_axes.assert_called_once_with([axes])
        labels = [call.kwargs["label"] for call in axes.errorbar.call_args_list]
        self.assertEqual(axes.legend.call_args.args, (handles, labels))
        axes.set_ylim.assert_not_called()
        self.assertEqual(len(labels), len(analysis.BACKENDS))
        self.assertTrue(all("degree=" in label for label in labels))
        self.assertTrue(all("batch=" in label for label in labels))
        self.assertTrue(any(label.startswith("BlockSelect") for label in labels))
        axes.set_title.assert_called_once()

    def test_cli_automatically_requests_both_versions_and_one_summary(self):
        self.summarize(self.comparison())
        output = self.path.parent / "plots"
        with patch.object(
            sys, "argv", [str(SCRIPT), str(self.path), "--output-dir", str(output)]
        ), patch.object(analysis, "plot") as plot:
            analysis.main()
        self.assertEqual(
            [call.kwargs["paper"] for call in plot.call_args_list], [False, True]
        )
        with (output / self.path.name).open() as source:
            summary = list(csv.DictReader(source))
        self.assertEqual({row["backend"] for row in summary}, set(analysis.BACKENDS))
        self.assertTrue(all("speedup_vs_air" in row for row in summary))

    def test_cli_prevalidates_paper_before_creating_outputs(self):
        self.summarize(self.comparison() + self.measurements(degree=16))
        output = self.path.parent / "plots"
        with patch.object(
            sys, "argv", [str(SCRIPT), str(self.path), "--output-dir", str(output)]
        ), patch.object(analysis, "plot") as plot, patch.object(
            sys, "stderr"
        ), self.assertRaises(SystemExit):
            analysis.main()
        plot.assert_not_called()
        self.assertFalse(output.exists())

    def test_cli_protects_raw_timings_and_detects_paper_path_collisions(self):
        self.summarize(self.comparison())
        original = self.path.read_bytes()
        command = [sys.executable, "-B", str(SCRIPT), str(self.path)]
        run = subprocess.run(
            command + ["--output-dir", str(self.path.parent)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("overwrite an input timing file", run.stderr)
        self.assertEqual(self.path.read_bytes(), original)
        other = self.path.with_stem(self.path.stem + "-paper")
        other.write_bytes(original)
        output = self.path.parent / "plots"
        run = subprocess.run(
            command + [str(other), "--output-dir", str(output)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("output paths collide", run.stderr)
        self.assertFalse(output.exists())

    def test_rejects_truncated_duplicate_and_invalid_measurements(self):
        cases = [self.rows[:-1], self.rows + [self.rows[1]], self.rows[1:]]
        for name, value in [
            ("seconds", "nan"),
            ("seconds", -1),
            ("retention_ratio", 1),
            ("iteration", -1),
            ("backend", "unknown"),
        ]:
            rows = copy.deepcopy(self.rows)
            rows[1][name] = value
            cases.append(rows)
        # All remaining phases agree, but iteration zero is missing.
        cases.append([row for row in self.rows if row["iteration"] != 0])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.summarize(rows)

    def test_rejects_unequal_repetition_counts(self):
        shorter = [row for row in self.measurements(degree=16) if row["iteration"] != 2]
        with self.assertRaisesRegex(ValueError, "repetition counts"):
            self.summarize(self.rows + shorter)

    def test_rejects_knn_csv_and_empty_input(self):
        for content in [
            "algorithm,time\nbits,0.1\n",
            "",
            ",".join(self.rows[0]) + "\n",
        ]:
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                analysis.summarize(self.path)

    def test_discovery_skips_empty_failed_runs_but_explicit_input_fails(self):
        root = Path(self.temp.name)
        (root / "data").mkdir()
        failed = root / "data" / self.path.name
        failed.touch()
        command = [sys.executable, "-B", str(SCRIPT)]
        run = subprocess.run(command, cwd=root, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Skipping empty run:", run.stderr)
        run = subprocess.run(
            command + [str(failed)], cwd=root, capture_output=True, text=True
        )
        self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
