"""Tensor timing validation and aggregation need only the Python standard library."""

import contextlib
import copy
import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "tensor_analysis", SCRIPTS / "tensor_analysis.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class TensorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "token-sampling-host-1.csv"

    def measurements(
        self,
        operator="token-sampling",
        backend="bits-sq",
        multiplier=1,
        phase_factors=None,
        **settings,
    ):
        config = (
            dict(
                operator=operator,
                dataset_id="a" * 64,
                backend=backend,
                rows=4096,
                batch_size=2 if operator == "token-sampling" else 1,
                k=32,
                retention_ratio=32 / 4096,
                degree=32 if backend == "bits-sq" else 1,
                block_size=512,
                items_per_thread=4,
                temperature=0.8 if operator == "token-sampling" else 0,
                seed=42 if operator == "token-sampling" else 0,
            )
            | settings
        )
        config["retention_ratio"] = config["k"] / config["rows"]
        rows = [config | dict(iteration=-1, phase="upload_shared", seconds=0.1)]
        for iteration, seconds in enumerate([0.001, 0.003, 0.009]):
            for phase in sorted(analysis.MEASURED_PHASES):
                factor = (phase_factors or {}).get(phase, 1)
                rows.append(
                    config
                    | dict(
                        iteration=iteration,
                        phase=phase,
                        seconds=seconds * multiplier * factor,
                    )
                )
        return rows

    def write(self, rows, path=None):
        path = path or self.path
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def summarize(self, rows, operator="token-sampling"):
        return analysis.summarize(self.write(rows), operator)

    def test_retains_every_measured_iteration_and_separate_phase(self):
        for operator in analysis.OPERATORS:
            summary = self.summarize(self.measurements(operator), operator)
            self.assertEqual(len(summary), 6)
            for row in summary:
                self.assertEqual(row["retention_ratio"], 32 / 4096)
                self.assertEqual(row["source_file"], str(self.path))
                self.assertEqual(row["speedup_vs_air"], "")
                if row["phase"] == "upload_shared":
                    self.assertEqual(row["samples"], 1)
                    self.assertEqual(row["median_ms"], 100)
                    self.assertEqual((row["p25_ms"], row["p75_ms"]), (100, 100))
                else:
                    self.assertEqual(row["samples"], 3)
                    self.assertEqual(row["median_ms"], 3)
                    self.assertEqual((row["p25_ms"], row["p75_ms"]), (2, 6))

    def test_keeps_datasets_shapes_temperatures_seeds_and_configurations_separate(self):
        variants = [
            {},
            {"degree": 16},
            {"block_size": 256},
            {"items_per_thread": 7},
            {"temperature": 0.7},
            {"seed": 43},
            {"k": 64},
            {"dataset_id": "b" * 64, "batch_size": 4},
            {"dataset_id": "c" * 64, "rows": 8192},
        ]
        rows = []
        for i, settings in enumerate(variants, 1):
            rows.extend(self.measurements(multiplier=i, **settings))
        actual = [row for row in self.summarize(rows) if row["phase"] == "operator"]
        self.assertEqual(len(actual), len(variants))
        for value, expected in zip(
            sorted(row["median_ms"] for row in actual),
            range(3, 3 * (len(variants) + 1), 3),
        ):
            self.assertAlmostEqual(value, expected)

    def test_speedups_use_matching_phase_and_include_losing_block_select(self):
        rows = self.measurements(phase_factors={"selection_isolated": 2})
        rows += self.measurements(
            backend="air-topk", multiplier=2, phase_factors={"selection_isolated": 0.5}
        )
        rows += self.measurements(backend="block-select", multiplier=4)
        rows += self.measurements(backend="bits", multiplier=3)
        rows += self.measurements(backend="bits-prefetch", multiplier=2)
        rows += self.measurements(backend="grid-select", multiplier=1)
        summary = self.summarize(rows)
        self.assertEqual({row["backend"] for row in summary}, set(analysis.BACKENDS))
        values = {(row["backend"], row["phase"]): row for row in summary}
        self.assertEqual(values["bits-sq", "operator"]["speedup_vs_air"], 2)
        self.assertEqual(values["bits-sq", "selection_isolated"]["speedup_vs_air"], 0.5)
        self.assertEqual(values["block-select", "operator"]["speedup_vs_air"], 0.5)
        self.assertEqual(values["air-topk", "operator"]["speedup_vs_air"], 1)

    def test_does_not_choose_best_bits_configuration(self):
        rows = self.measurements(degree=16, multiplier=2)
        rows += self.measurements(degree=32, multiplier=1)
        rows += self.measurements(backend="air-topk", multiplier=3)
        summary = [
            row
            for row in self.summarize(rows)
            if row["phase"] == "operator" and row["backend"] == "bits-sq"
        ]
        self.assertEqual(len(summary), 2)
        for actual, expected in zip(
            sorted(row["speedup_vs_air"] for row in summary), (1.5, 3)
        ):
            self.assertAlmostEqual(actual, expected)

    def test_absent_or_ambiguous_air_baseline_has_no_speedup(self):
        for changed in (
            {"temperature": 0.7},
            {"seed": 43},
            {"k": 64},
            {"dataset_id": "b" * 64},
            {"dataset_id": "b" * 64, "batch_size": 1},
        ):
            with self.subTest(changed=changed):
                rows = self.measurements() + self.measurements(
                    backend="air-topk", **changed
                )
                summary = self.summarize(rows)
                self.assertTrue(
                    all(
                        row["speedup_vs_air"] == ""
                        for row in summary
                        if row["backend"] == "bits-sq"
                    )
                )
        rows = self.measurements() + self.measurements(
            backend="air-topk", block_size=256
        )
        rows += self.measurements(backend="air-topk", block_size=512)
        summary = self.summarize(rows)
        self.assertEqual(len(summary), 18)
        self.assertTrue(all(row["speedup_vs_air"] == "" for row in summary))

    def test_files_do_not_share_baselines_or_samples(self):
        first = analysis.summarize(self.write(self.measurements()), "token-sampling")
        second_path = self.root / "token-sampling-othergpu-2.csv"
        second = analysis.summarize(
            self.write(self.measurements(backend="air-topk"), second_path),
            "token-sampling",
        )
        self.assertTrue(all(row["speedup_vs_air"] == "" for row in first))
        self.assertTrue(all(row["speedup_vs_air"] == 1 for row in second))
        self.assertNotEqual(first[0]["source_file"], second[0]["source_file"])

    def test_plots_require_matching_unique_air_baselines(self):
        cases = [
            self.measurements(),
            self.measurements() + self.measurements(backend="air-topk", k=64),
            self.measurements()
            + self.measurements(backend="air-topk", block_size=256)
            + self.measurements(backend="air-topk", block_size=512),
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(
                ValueError, "unique AIR Top-K baseline"
            ):
                analysis.plot_pages(self.summarize(rows), self.path, paper=True)

    def test_plot_preparation_retains_block_select_and_separate_workloads(self):
        rows = []
        for seed in (42, 43):
            for k in (32, 64):
                for backend in analysis.BACKENDS:
                    rows += self.measurements(
                        backend=backend,
                        seed=seed,
                        k=k,
                        items_per_thread=(k // 32 if backend == "block-select" else 4),
                    )
        summary = self.summarize(rows)
        original = copy.deepcopy(summary)
        pages = analysis.plot_pages(summary, self.path)
        self.assertEqual(len(pages), 2)
        for points in pages.values():
            self.assertEqual({row["backend"] for row in points}, set(analysis.BACKENDS))
            self.assertEqual(
                {row["phase"] for row in points}, {"operator", "selection_isolated"}
            )
        self.assertEqual(summary, original)

    def test_paper_rejects_baseline_configuration_changes_even_at_different_k(self):
        for k in (32, 64):
            rows = self.measurements(backend="grid-select", block_size=128)
            rows += self.measurements(backend="grid-select", block_size=256, k=k)
            for air_k in {32, k}:
                rows += self.measurements(backend="air-topk", k=air_k)
            with self.subTest(k=k), self.assertRaisesRegex(
                ValueError, "one fixed configuration per backend"
            ):
                analysis.plot_pages(self.summarize(rows), self.path, paper=True)

    def test_paper_pair_choice_uses_operator_medians_for_both_panels(self):
        for operator in analysis.OPERATORS:
            rows = []
            for k in (32, 64):
                rows += self.measurements(
                    operator, backend="air-topk", k=k, multiplier=4
                )
                rows += self.measurements(operator, backend="block-select", k=k)
                for block in (128, 256, 512):
                    for items in (4, 13):
                        winner = (block, items) == ((128, 13) if k == 32 else (512, 4))
                        rows += self.measurements(
                            operator,
                            block_size=block,
                            items_per_thread=items,
                            k=k,
                            phase_factors={
                                "operator": 1 if winner else 2,
                                "selection_isolated": 3 if winner else 1,
                            },
                        )
            summary = self.summarize(rows, operator)
            original = copy.deepcopy(summary)
            detailed = analysis.plot_pages(summary, self.path)
            paper = analysis.plot_pages(summary, self.path, paper=True)
            self.assertEqual(sum(map(len, detailed.values())), 32)
            self.assertEqual(sum(map(len, paper.values())), 8)
            for values in paper.values():
                for row in values:
                    if row["backend"] == "bits-sq":
                        self.assertEqual(
                            (row["block_size"], row["items_per_thread"]),
                            (128, 13) if row["k"] == 32 else (512, 4),
                        )
                        self.assertAlmostEqual(
                            row["median_ms"], 3 if row["phase"] == "operator" else 9
                        )
            self.assertEqual(summary, original)

    def test_global_paper_uses_one_pair_per_backend_and_operator_geometric_mean(self):
        for operator in analysis.OPERATORS:
            with self.subTest(operator=operator):
                rows = []
                factors = {
                    "bits-sq": {
                        (128, 4): (1, 9),
                        (128, 13): (4, 4),
                        (256, 7): (2, 6),
                    },
                    "bits-prefetch": {
                        (128, 4): (4, 4),
                        (128, 13): (1, 9),
                        (256, 7): (2, 6),
                    },
                }
                for position, k in enumerate((32, 64)):
                    rows += self.measurements(
                        operator, backend="air-topk", k=k, multiplier=12
                    )
                    rows += self.measurements(operator, backend="block-select", k=k)
                    for backend, pairs in factors.items():
                        for (block, items), operator_factors in pairs.items():
                            factor = operator_factors[position]
                            rows += self.measurements(
                                operator,
                                backend=backend,
                                block_size=block,
                                items_per_thread=items,
                                k=k,
                                phase_factors={
                                    "operator": factor,
                                    "selection_isolated": 12 / factor,
                                },
                            )
                summary = self.summarize(rows, operator)
                original = copy.deepcopy(summary)
                per_k = next(
                    iter(analysis.plot_pages(summary, self.path, paper=True).values())
                )
                global_rows = next(
                    iter(
                        analysis.plot_pages(
                            summary, self.path, paper=True, selection="global"
                        ).values()
                    )
                )
                winners = {"bits-sq": (128, 4), "bits-prefetch": (128, 13)}
                per_k_winners = {
                    ("bits-sq", 32): (128, 4),
                    ("bits-sq", 64): (128, 13),
                    ("bits-prefetch", 32): (128, 13),
                    ("bits-prefetch", 64): (128, 4),
                }
                for backend, winner in winners.items():
                    self.assertEqual(
                        {
                            (row["block_size"], row["items_per_thread"])
                            for row in per_k
                            if row["backend"] == backend
                        },
                        {(128, 4), (128, 13)},
                    )
                    selected = [row for row in global_rows if row["backend"] == backend]
                    self.assertEqual(len(selected), 4)
                    self.assertEqual(
                        {
                            (row["block_size"], row["items_per_thread"])
                            for row in selected
                        },
                        {winner},
                    )
                    for row in selected:
                        factor = 1 if row["k"] == 32 else 9
                        expected = 3 * (
                            factor if row["phase"] == "operator" else 12 / factor
                        )
                        self.assertAlmostEqual(row["median_ms"], expected)
                self.assertEqual(summary, original)
                output = self.root / "plots" / operator
                with patch.object(
                    sys,
                    "argv",
                    ["plot", str(self.path), "--output-dir", str(output)],
                ), patch.object(analysis, "plot"):
                    analysis.main(operator)
                with (output / self.path.name).open() as source:
                    exported = list(csv.DictReader(source))
                self.assertEqual(len(exported), len(summary))
                for row in exported:
                    backend = row["backend"]
                    pair = (int(row["block_size"]), int(row["items_per_thread"]))
                    self.assertEqual(
                        row["paper_global_selected"],
                        str(backend == "air-topk" or pair == winners.get(backend)),
                    )
                    self.assertEqual(
                        row["paper_selected"],
                        str(
                            backend == "air-topk"
                            or pair == per_k_winners.get((backend, int(row["k"])))
                        ),
                    )

    def test_global_paper_excludes_pairs_without_complete_k_coverage(self):
        rows = self.measurements(block_size=128, items_per_thread=4, multiplier=0.1)
        rows += self.measurements(
            block_size=128, items_per_thread=13, k=64, multiplier=0.1
        )
        for k in (32, 64):
            rows += self.measurements(backend="air-topk", k=k)
            rows += self.measurements(
                block_size=256, items_per_thread=7, k=k, multiplier=2
            )
        summary = self.summarize(rows)
        pages = analysis.plot_pages(summary, self.path, paper=True, selection="global")
        selected = [
            row
            for points in pages.values()
            for row in points
            if row["backend"] == "bits-sq"
        ]
        self.assertEqual(
            {(row["block_size"], row["items_per_thread"]) for row in selected},
            {(256, 7)},
        )
        self.assertEqual({row["k"] for row in selected}, {32, 64})

    def test_rejects_duplicates_missing_phases_uploads_and_iteration_gaps(self):
        rows = self.measurements()
        cases = [rows[:-1], rows + [rows[1]], rows[1:], rows + [rows[0]]]
        cases.append([row for row in rows if row["iteration"] != 0])
        cases.append([row for row in rows if row["phase"] != "output_isolated"])
        cases.append([row for row in rows if row["phase"] == "upload_shared"])
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.summarize(case)

    def test_rejects_unequal_repetition_counts(self):
        rows = self.measurements()
        shorter = [
            row
            for row in self.measurements(backend="air-topk")
            if row["iteration"] != 2
        ]
        with self.assertRaisesRegex(ValueError, "repetition counts"):
            self.summarize(rows + shorter)

    def test_rejects_invalid_values_and_unknown_backends(self):
        changes = [
            ("seconds", "nan"),
            ("seconds", "inf"),
            ("seconds", 0),
            ("seconds", -1),
            ("retention_ratio", 1),
            ("retention_ratio", "nan"),
            ("iteration", -1),
            ("iteration", "1.5"),
            ("phase", "warmup"),
            ("backend", "unknown"),
            ("operator", "gradient-compression"),
            ("dataset_id", "not-a-digest"),
            ("dataset_id", "A" * 64),
            ("rows", 0),
            ("rows", 2**31),
            ("batch_size", 0),
            ("batch_size", 2**31),
            ("k", 0),
            ("k", 2049),
            ("degree", 0),
            ("degree", 4097),
            ("block_size", -1),
            ("items_per_thread", -1),
            ("temperature", 0),
            ("temperature", -1),
            ("temperature", "nan"),
            ("temperature", "inf"),
            ("seed", -1),
            ("seed", 2**64),
        ]
        for name, value in changes:
            rows = copy.deepcopy(self.measurements())
            rows[1][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.summarize(rows)
        rows = self.measurements()
        rows[0]["iteration"] = 0
        with self.assertRaisesRegex(ValueError, "iteration -1"):
            self.summarize(rows)

    def test_gradient_settings_and_dataset_shapes_must_be_consistent(self):
        for settings in ({"batch_size": 2}, {"temperature": 1}, {"seed": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.summarize(
                    self.measurements("gradient-compression", **settings),
                    "gradient-compression",
                )
        for settings in ({"rows": 8192}, {"batch_size": 3}):
            with self.subTest(settings=settings), self.assertRaisesRegex(
                ValueError, "input shapes"
            ):
                self.summarize(self.measurements() + self.measurements(**settings))

    def test_rejects_empty_wrong_and_malformed_csvs(self):
        valid = self.write(self.measurements()).read_text()
        header, body = valid.split("\n", 1)
        cases = [
            "",
            "algorithm,time\nbits,0.1\n",
            header + "\n",
            header + ",seconds\n" + body,
            header + "\n" + body.split("\n", 1)[0] + ",extra\n",
            header + "\n" + body.split("\n", 1)[0].rsplit(",", 1)[0] + "\n",
        ]
        for content in cases:
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                analysis.summarize(self.path, "token-sampling")
        with self.assertRaisesRegex(ValueError, "Unknown tensor operator"):
            analysis.summarize(self.path, "database-topn")

    def test_discovery_skips_only_empty_failed_jobs(self):
        data = self.root / "data"
        data.mkdir()
        failed = data / "token-sampling-host-1.csv"
        failed.touch()
        valid = self.write(self.measurements(), data / "token-sampling-host-2.csv")
        unrelated = data / "database-topn-host-1.csv"
        unrelated.write_text("not tensor timings")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(analysis.discover_files("token-sampling", data), [valid])
        self.assertIn(f"Skipping empty run: {failed}", stderr.getvalue())

    def test_cli_empty_discovery_succeeds_but_explicit_empty_input_fails(self):
        data = self.root / "data"
        data.mkdir()
        for operator in analysis.OPERATORS:
            failed = data / f"{operator}-host-1.csv"
            failed.touch()
            command = [sys.executable, "-B", str(SCRIPTS / f"plot-{operator}.py")]
            run = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Skipping empty run:", run.stderr)
            self.assertIn("No " + operator, run.stderr)
            run = subprocess.run(
                command + [str(failed)], cwd=self.root, capture_output=True, text=True
            )
            self.assertNotEqual(run.returncode, 0)

    def test_cli_rejects_output_name_collisions_before_plotting(self):
        first = self.write(self.measurements())
        folder = self.root / "other-host"
        folder.mkdir()
        second = self.write(self.measurements(), folder / first.name)
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS / "plot-token-sampling.py"),
                str(first),
                str(second),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("distinct basenames", run.stderr)

    def test_cli_cannot_overwrite_raw_timings_in_output_directory(self):
        for operator in analysis.OPERATORS:
            with self.subTest(operator=operator):
                path = self.root / f"{operator}-host-1.csv"
                self.write(self.measurements(operator), path)
                original = path.read_bytes()
                run = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(SCRIPTS / f"plot-{operator}.py"),
                        str(path),
                        "--output-dir",
                        str(self.root),
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("overwrite an input timing file", run.stderr)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(path.with_suffix(".pdf").exists())

    def test_cli_rejects_detailed_and_paper_filename_collisions(self):
        first = self.write(self.measurements())
        output = self.root / "plots"
        for suffix in ("-paper", "-paper-global"):
            with self.subTest(suffix=suffix):
                second = self.write(
                    self.measurements(), self.root / f"{first.stem}{suffix}.csv"
                )
                run = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(SCRIPTS / "plot-token-sampling.py"),
                        str(first),
                        str(second),
                        "--output-dir",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("output filenames collide", run.stderr)
                self.assertFalse(output.exists())

    def test_cli_validates_global_coverage_before_writing_any_outputs(self):
        rows = self.measurements(block_size=128, items_per_thread=4, k=32)
        rows += self.measurements(block_size=128, items_per_thread=13, k=64)
        for k in (32, 64):
            rows += self.measurements(backend="air-topk", k=k)
        self.write(rows)
        output = self.root / "plots"
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS / "plot-token-sampling.py"),
                str(self.path),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("global paper selection", run.stderr)
        self.assertIn("bits-sq", run.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
