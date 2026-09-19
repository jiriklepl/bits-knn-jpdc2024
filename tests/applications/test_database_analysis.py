"""Check database timing aggregation without requiring plotting packages or a GPU."""

import copy
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot-database-topn.py"
SPEC = importlib.util.spec_from_file_location("database_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "database-topn-host-1.csv"
        self.rows = self.measurements()

    def measurements(self, digest="a" * 64, degree=32, multiplier=1):
        config = dict(
            dataset_id=digest,
            backend="bits-sq",
            rows=4096,
            k=32,
            retention_ratio=32 / 4096,
            degree=degree,
            block_size=512,
            items_per_thread=4,
        )
        rows = [config | dict(iteration=-1, phase="upload_shared", seconds=0.1)]
        for iteration, seconds in enumerate([0.001, 0.003, 0.009]):
            for phase in sorted(analysis.MEASURED_PHASES):
                rows.append(
                    config
                    | dict(
                        iteration=iteration, phase=phase, seconds=seconds * multiplier
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

    def test_rejects_truncated_duplicate_and_invalid_measurements(self):
        cases = [self.rows[:-1], self.rows + [self.rows[1]], self.rows[1:]]
        for name, value in [
            ("seconds", "nan"),
            ("seconds", -1),
            ("retention_ratio", 1),
            ("iteration", -1),
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
