"""Checks for the report's averaging, phase pairing, and fixed-parameter selection."""

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "report_manuscript", Path(__file__).resolve().parents[1] / "scripts/report-manuscript.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def sample(**changes):
    return dict(point_count=1024, query_count=256, dim=1, k=32,
                algorithm="bits", generator="uniform", preprocessor="identity",
                block_size=128, items_per_thread="1,1,1", deg=1,
                iteration=10, phase="selection", time=1.0) | changes


class ManuscriptReportTests(unittest.TestCase):
    def test_warmup_and_tuning_repetitions(self):
        rows = [sample(iteration=i, time=999 if i < 10 else (2 if i < 20 else 6))
                for i in range(30)]
        frame = pd.DataFrame(rows)
        self.assertEqual(report.summarize(frame, ["selection"], 20).time.iloc[0], 2)
        self.assertEqual(report.summarize(frame, ["selection"], 30).time.iloc[0], 4)

    def test_speedup_uses_ratio_of_mean_times(self):
        rows = [sample(algorithm=alg, iteration=i, time=time)
                for i in range(10, 20)
                for alg, time in [("baseline", 10), ("bits", 1 if i < 15 else 9)]]
        means = report.summarize(pd.DataFrame(rows), ["selection"], 20)
        speed = report.ratio(report.matrix(means), "baseline", "bits")
        self.assertEqual(speed.iloc[0], 2)

    def test_phases_are_added_before_averaging(self):
        rows = [sample(iteration=i, phase=phase, time=time)
                for i in range(10, 20)
                for phase, time in [("distances", 3), ("selection", 2)]]
        means = report.summarize(pd.DataFrame(rows), ["distances", "selection"], 20)
        self.assertEqual(means.time.iloc[0], 5)
        with self.assertRaisesRegex(ValueError, "phase pair"):
            report.summarize(pd.DataFrame(rows[:-1]), ["distances", "selection"], 20)

    def test_duplicate_and_missing_iterations_fail(self):
        rows = [sample(iteration=i) for i in range(10, 20)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report.summarize(pd.DataFrame(rows + [rows[0]]), ["selection"], 20)
        with self.assertRaisesRegex(ValueError, "repetitions"):
            report.summarize(pd.DataFrame(rows[:-1]), ["selection"], 20)

    def test_incomplete_tuple_cannot_win_fixed_tuning(self):
        # Block 512 wins one case but is absent from the second case.
        frame = pd.DataFrame([sample(k=k, block_size=b, time=t)
                              for k, b, t in [(32, 128, 2), (64, 128, 2),
                                              (32, 256, 1), (64, 256, 10),
                                              (32, 512, 0.5)]])
        fixed = report.fixed_parameters(frame)
        self.assertEqual(fixed.block_size.tolist(), [128, 128])
        self.assertEqual(fixed.slowdown.max(), 4)

    def test_unsupported_comparisons_are_not_counted_as_wins(self):
        frame = pd.DataFrame([sample(algorithm="raft", k=32, time=2),
                              sample(k=32), sample(k=128)])
        speed = report.ratio(report.matrix(frame), "raft", "bits")
        self.assertEqual(report.wins(speed), "1/1")


if __name__ == "__main__":
    unittest.main()
