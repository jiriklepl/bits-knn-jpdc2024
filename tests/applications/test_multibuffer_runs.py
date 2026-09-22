"""Buffer campaigns must not be pooled merely because they share a host."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
if PANDAS_AVAILABLE:
    import pandas as pd
    from multibuffer_runs import group_runs


@unittest.skipUnless(PANDAS_AVAILABLE, "Install plotting requirements")
class MultibufferGroupingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / "runs.json"
        self.register([])

    def register(self, groups):
        self.registry.write_text(json.dumps({"version": 1, "groups": groups}))

    def run_file(self, job, preprocessor, host="gpu01", **settings):
        path = self.root / f"buffer-{host}-{job}.csv"
        rows = [
            dict(
                algorithm=algorithm,
                generator="uniform",
                preprocessor=preprocessor,
                point_count=1024,
                query_count=64,
                dim=1,
                block_size=128,
                k=32,
                items_per_thread="1,1,1",
                deg=1,
                iteration=iteration,
                phase="selection",
                time=job / 1000,
            )
            | settings
            for algorithm in ("bits", "partial-bitonic-regs")
            for iteration in range(12)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def groups(self):
        return group_runs(self.root.glob("buffer-*.csv"), self.registry)

    def test_repeated_campaigns_stay_separate_even_with_identical_settings(self):
        for start in (100, 103, 800):
            for offset, variant in enumerate(("identity", "ascending", "descending")):
                self.run_file(start + offset, variant)
        groups, warnings = self.groups()
        self.assertEqual(warnings, [])
        self.assertEqual(
            [[r.jobid for r in runs] for _, runs in groups],
            [[100, 101, 102], [103, 104, 105], [800, 801, 802]],
        )
        self.assertEqual(len({name for name, _ in groups}), 3)

    def test_registry_handles_interleaved_legacy_jobs_and_preserves_paper_name(self):
        self.register(
            [dict(hostname="gpu01", jobs=[10, 14, 19], output="multibuffer-gpu01")]
        )
        for job, variant in zip((10, 14, 19), ("ascending", "descending", "identity")):
            self.run_file(job, variant)
        groups, warnings = self.groups()
        self.assertEqual(warnings, [])
        self.assertEqual([name for name, _ in groups], ["multibuffer-gpu01"])
        self.assertEqual([r.jobid for r in groups[0][1]], [10, 14, 19])

    def test_missing_known_member_cannot_be_replaced_by_a_nearby_rerun(self):
        self.register(
            [dict(hostname="gpu01", jobs=[100, 104, 109], output="multibuffer-gpu01")]
        )
        for job, variant in (
            (100, "identity"),
            (101, "ascending"),
            (102, "descending"),
            (104, "ascending"),
        ):
            self.run_file(job, variant)
        groups, warnings = self.groups()
        self.assertEqual(groups, [])
        self.assertTrue(any("known group" in warning for warning in warnings))

    def test_heuristic_rejects_gaps_different_hosts_and_incompatible_measurements(self):
        for difference in (
            "gap",
            "host",
            "k",
            "block_size",
            "iteration",
            "generator",
            "duplicate",
        ):
            with self.subTest(difference=difference):
                for path in self.root.glob("buffer-*.csv"):
                    path.unlink()
                self.run_file(100, "identity")
                self.run_file(101, "ascending")
                settings = (
                    {
                        difference: {
                            "k": 64,
                            "block_size": 256,
                            "iteration": 1,
                            "generator": "normal",
                        }[difference]
                    }
                    if difference in ("k", "block_size", "iteration", "generator")
                    else {}
                )
                self.run_file(
                    103 if difference == "gap" else 102,
                    "ascending" if difference == "duplicate" else "descending",
                    host="gpu02" if difference == "host" else "gpu01",
                    **settings,
                )
                groups, warnings = self.groups()
                self.assertEqual(groups, [])
                self.assertTrue(warnings)

    def test_registered_groups_still_require_matching_measurement_layouts(self):
        self.register(
            [dict(hostname="gpu01", jobs=[100, 101, 102], output="multibuffer-gpu01")]
        )
        self.run_file(100, "identity")
        self.run_file(101, "ascending")
        self.run_file(102, "descending", k=64)
        groups, warnings = self.groups()
        self.assertEqual(groups, [])
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
