"""Exercise nine-case application studies, provenance checks, and original plots."""

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import application_scaling_analysis as analysis  # noqa: E402


class ScalingFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "index.json"
        self.index = self.make_study()
        self.save_index()

    def save_index(self, *, update_hash=True):
        if update_hash:
            self.index["settings_sha256"] = hashlib.sha256(
                json.dumps(self.index["settings"], sort_keys=True).encode()
            ).hexdigest()
        self.path.write_text(json.dumps(self.index))

    def write_raw(self, run, rows):
        path = self.root / run["csv"]
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        run["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def raw(self, run):
        with (self.root / run["csv"]).open(newline="") as source:
            return list(csv.DictReader(source))

    def make_study(self, *, tiers=("small", "middle", "large")):
        settings = dict(
            ks=[32, 64],
            degrees=[8, 32],
            blocks=[128, 512],
            items=[4, 8],
            repeat=3,
            warmup=1,
        )
        index = dict(version=1, settings=settings, workloads=[], runs=[])
        for operator in analysis.OPERATORS:
            for position, suffix in enumerate(tiers):
                identity = f"{operator}-{suffix}"
                sizes = (512, 1024, 2048)
                batches = (2, 4, 8)
                candidates = 512 if operator == "token-sampling" else sizes[position]
                batch = batches[position] if operator == "token-sampling" else 1
                workload = dict(
                    id=identity,
                    operator=operator,
                    label=f"{operator} {suffix}",
                    size_tier=suffix,
                    dataset_id=hashlib.sha256(identity.encode()).hexdigest(),
                    size=batch if operator == "token-sampling" else candidates,
                    score_bytes=4 * candidates * batch,
                )
                if operator == "database-topn":
                    workload.update(rows=candidates, size_unit="rows")
                elif operator == "token-sampling":
                    workload.update(
                        batch_size=batch, vocabulary_size=candidates, size_unit="batch"
                    )
                else:
                    workload.update(elements=candidates, size_unit="elements")
                index["workloads"].append(workload)
                ks = settings["ks"]
                run = dict(
                    workload_id=identity,
                    mode="fixed-k",
                    ks=ks,
                    csv=f"{identity}.csv",
                    status="complete",
                )
                configurations, rows = [], []
                for k in ks:
                    configurations += [
                        (backend, k, degree, block, items)
                        for backend, degrees in (
                            ("bits-prefetch", (1,)),
                            ("bits-sq", (8, 32)),
                        )
                        for degree in degrees
                        for block in (128, 512)
                        for items in (4, 8)
                    ]
                    configurations += [
                        ("air-topk", k, 1, 512, 0),
                        ("grid-select", k, 1, 0, 0),
                    ]
                    if k in (32, 64):
                        configurations.append(
                            ("block-select", k, 1, 128, 2 if k == 32 else 3)
                        )
                for backend, k, degree, block, items in configurations:
                    latency = 20.0
                    if backend == "bits-sq":
                        if (degree, block, items) == (8, 128, 4):
                            latency = 1.0 if suffix == "small" else 9.0
                        elif (degree, block, items) == (32, 512, 8):
                            latency = 4.0
                    elif backend == "bits-prefetch":
                        if (block, items) == (128, 4):
                            latency = 2.0 if suffix == "small" else 8.0
                        elif (block, items) == (512, 8):
                            latency = 5.0
                    config = dict(
                        dataset_id=workload["dataset_id"],
                        backend=backend,
                        rows=candidates,
                        k=k,
                        retention_ratio=k / candidates,
                        degree=degree,
                        block_size=block,
                        items_per_thread=items,
                    )
                    if operator != "database-topn":
                        config.update(
                            operator=operator,
                            batch_size=batch,
                            temperature=1 if operator == "token-sampling" else 0,
                            seed=42 if operator == "token-sampling" else 0,
                        )
                    rows.append(
                        config
                        | dict(iteration=-1, phase="upload_shared", seconds=0.1)
                    )
                    for iteration, factor in enumerate((0.8, 1, 1.2)):
                        for phase in (
                            "operator",
                            "selection_isolated",
                            "transform_isolated",
                            "output_isolated",
                            "download",
                        ):
                            value = (
                                20 / latency
                                if phase == "selection_isolated"
                                else latency
                            )
                            rows.append(
                                config
                                | dict(
                                    iteration=iteration,
                                    phase=phase,
                                    seconds=value * factor / 1000,
                                )
                            )
                run["configurations"] = len(configurations)
                self.write_raw(run, rows)
                index["runs"].append(run)
        return index


class ScalingAnalysisTests(ScalingFixture):
    def cli(self):
        spec = importlib.util.spec_from_file_location(
            "plot_application_scaling", SCRIPTS / "plot-application-scaling.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        return cli

    def test_default_dispatches_all_nine_cases_to_original_three_version_plotters(self):
        cli = self.cli()
        output = self.root / "plots"
        args = [cli.__file__, str(self.path), "--output-dir", str(output)]
        with patch.object(sys, "argv", args), patch.object(
            sys, "stderr", io.StringIO()
        ), patch.object(cli.subprocess, "run") as execute:
            cli.main()
        self.assertEqual(execute.call_count, 9)
        commands = [call.args[0] for call in execute.call_args_list]
        cases = {
            f"{operator}-{tier}"
            for operator in analysis.OPERATORS
            for tier in ("small", "middle", "large")
        }
        self.assertEqual({Path(command[2]).stem for command in commands}, cases)
        for command in commands:
            operator = next(
                name
                for name in analysis.OPERATORS
                if Path(command[2]).stem.startswith(name)
            )
            self.assertEqual(
                command[:2], [sys.executable, str(SCRIPTS / f"plot-{operator}.py")]
            )
            self.assertEqual(command[-2:], ["--output-dir", str(output)])
        self.assertTrue(all(call.kwargs["check"] for call in execute.call_args_list))

    def test_default_requires_every_size_tier_for_each_application(self):
        original = copy.deepcopy(self.index)
        for failure in (
            "missing_case",
            "missing_tier",
            "duplicate_tier",
            "unknown_tier",
            "filename",
        ):
            with self.subTest(failure=failure):
                self.index = copy.deepcopy(original)
                workload = self.index["workloads"][1]
                if failure == "missing_case":
                    self.index["workloads"].pop()
                    self.index["runs"].pop()
                elif failure == "missing_tier":
                    del workload["size_tier"]
                elif failure == "duplicate_tier":
                    workload["size_tier"] = "small"
                elif failure == "unknown_tier":
                    workload["size_tier"] = "medium"
                else:
                    run = self.index["runs"][0]
                    rows = self.raw(run)
                    run["csv"] = "unlabeled.csv"
                    self.write_raw(run, rows)
                self.save_index()
                self.reject_before_outputs(
                    "CSV names"
                    if failure == "filename"
                    else "exactly nine fixed-k cases"
                )

    def test_removed_plot_modes_are_rejected_before_loading(self):
        cli = self.cli()
        for option in ("--cross-size", "--skip-individual-runs"):
            with self.subTest(option=option), patch.object(
                sys, "argv", [cli.__file__, str(self.path), option]
            ), patch.object(sys, "stderr", io.StringIO()) as stderr, patch.object(
                cli, "load_study"
            ) as load, self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(f"unrecognized arguments: {option}", stderr.getvalue())
            load.assert_not_called()

    def test_verified_study_keeps_each_case_and_its_measurements_separate(self):
        rows = analysis.load_study(self.path)
        workloads = {item["id"]: item for item in self.index["workloads"]}
        self.assertEqual({row["workload_id"] for row in rows}, set(workloads))
        self.assertEqual(len(rows), 9 * 30 * 6)
        for row in rows:
            workload = workloads[row["workload_id"]]
            self.assertEqual(row["operator"], workload["operator"])
            self.assertEqual(row["size_tier"], workload["size_tier"])
            self.assertEqual(row["dataset_id"], workload["dataset_id"])
            self.assertEqual(row["mode"], "fixed-k")
            self.assertEqual(
                Path(row["source_csv"]), self.root / f"{workload['id']}.csv"
            )
            self.assertIn(row["k"], (32, 64))
            self.assertEqual(
                row["samples"], 1 if row["phase"] == "upload_shared" else 3
            )
            self.assertNotIn("scaling_global_selected", row)
            if (
                row["backend"] == "bits-sq"
                and row["degree"] == 8
                and row["block_size"] == 128
                and row["items_per_thread"] == 4
                and row["phase"] == "operator"
            ):
                self.assertAlmostEqual(
                    row["median_ms"], 1 if row["size_tier"] == "small" else 9
                )

    def test_saved_fixed_k_studies_accept_empty_legacy_retention_fields(self):
        expected = analysis.load_study(self.path)
        self.index["settings"]["retention_ratios"] = []
        for run in self.index["runs"]:
            run["requested_retention"] = None
        self.save_index()
        self.assertEqual(analysis.load_study(self.path), expected)

    def test_retention_studies_are_rejected_before_outputs(self):
        original = copy.deepcopy(self.index)
        for failure in ("ratios", "mode", "requested_retention"):
            with self.subTest(failure=failure):
                self.index = copy.deepcopy(original)
                if failure == "ratios":
                    self.index["settings"]["retention_ratios"] = [0.01]
                elif failure == "mode":
                    self.index["runs"][-1]["mode"] = "retention"
                else:
                    self.index["runs"][-1]["requested_retention"] = 0.01
                self.save_index()
                self.reject_before_outputs("support fixed-k runs only")

    def reject_before_outputs(self, expression):
        output = self.root / "plots"
        run = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "plot-application-scaling.py"),
                str(self.path),
                "--output-dir",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertRegex(run.stderr, expression)
        self.assertFalse(output.exists())

    def test_bad_provenance_and_incomplete_index_fail_before_any_outputs(self):
        original = copy.deepcopy(self.index)
        for failure, pattern in (
            ("checksum", "Timing CSV checksum"),
            ("settings_hash", "settings checksum"),
            ("dataset", "dataset does not match"),
            ("missing_run", "every workload"),
            ("failed", "Incomplete study"),
            ("duplicate", "Duplicate study run"),
            ("size", "dimensions are inconsistent"),
            ("planned_ks", "Planned k values"),
            ("repetitions", "repetition count"),
        ):
            with self.subTest(failure=failure):
                self.index = copy.deepcopy(original)
                if failure == "checksum":
                    self.index["runs"][0]["csv_sha256"] = "0" * 64
                elif failure == "settings_hash":
                    self.index["settings_sha256"] = "0" * 64
                elif failure == "dataset":
                    self.index["workloads"][0]["dataset_id"] = "0" * 64
                elif failure == "missing_run":
                    self.index["runs"].pop(0)
                elif failure == "failed":
                    self.index["runs"][0]["status"] = "failed"
                elif failure == "duplicate":
                    self.index["runs"].append(self.index["runs"][0])
                elif failure == "size":
                    self.index["workloads"][0]["size"] *= 2
                elif failure == "planned_ks":
                    self.index["runs"][0]["ks"] = [32]
                elif failure == "repetitions":
                    self.index["settings"]["repeat"] = 4
                self.save_index(update_hash=failure != "settings_hash")
                self.reject_before_outputs(pattern)

    def test_complete_configurations_required_even_when_row_count_and_hash_match(self):
        run = self.index["runs"][0]
        rows = self.raw(run)
        for row in rows:
            if (
                row["backend"] == "bits-sq"
                and row["degree"] == "32"
                and row["block_size"] == "512"
                and row["items_per_thread"] == "8"
            ):
                row["degree"] = "16"
        self.write_raw(run, rows)
        self.save_index()
        self.reject_before_outputs("do not cover the planned sweep")

    def test_missing_configuration_is_not_accepted_as_a_complete_study(self):
        run = self.index["runs"][0]
        rows = [
            row
            for row in self.raw(run)
            if not (
                row["backend"] == "bits-sq"
                and row["degree"] == "32"
                and row["k"] == "64"
            )
        ]
        self.write_raw(run, rows)
        self.save_index()
        self.reject_before_outputs("do not cover the planned sweep")

    def test_sampling_shape_includes_vocabulary_and_settings(self):
        original = copy.deepcopy(self.index)
        for field, value, pattern in (
            ("vocabulary_size", 1024, "Timing shape"),
            ("seed", 43, "Sampling settings"),
        ):
            with self.subTest(field=field):
                self.index = copy.deepcopy(original)
                workload = next(
                    item
                    for item in self.index["workloads"]
                    if item["operator"] == "token-sampling"
                )
                if field == "vocabulary_size":
                    workload[field] = value
                    workload["score_bytes"] *= 2
                else:
                    run = next(
                        item
                        for item in self.index["runs"]
                        if item["workload_id"] == workload["id"]
                    )
                    rows = self.raw(run)
                    for row in rows:
                        row[field] = value
                    self.write_raw(run, rows)
                self.save_index()
                self.reject_before_outputs(pattern)


PLOTTING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "matplotlib", "seaborn")
)


@unittest.skipUnless(
    PLOTTING_AVAILABLE, "Install plotting requirements for Agg render checks"
)
class ScalingRenderTests(ScalingFixture):
    def test_default_writes_27_original_pdfs_and_selects_globally_within_each_case(
        self
    ):
        output = self.root / "plots"
        spec = importlib.util.spec_from_file_location(
            "plot_application_scaling", SCRIPTS / "plot-application-scaling.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        def execute_original_plotter(command, *, check):
            self.assertTrue(check)
            with patch.object(sys, "argv", command[1:]):
                runpy.run_path(command[1], run_name="__main__")

        with patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / "cache")}):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            self.addCleanup(plt.close, "all")
            with patch.object(
                sys,
                "argv",
                [cli.__file__, str(self.path), "--output-dir", str(output)],
            ), patch.object(
                cli,
                "subprocess",
                SimpleNamespace(
                    run=execute_original_plotter,
                    CalledProcessError=subprocess.CalledProcessError,
                ),
            ):
                cli.main()
        cases = {
            f"{operator}-{tier}"
            for operator in analysis.OPERATORS
            for tier in ("small", "middle", "large")
        }
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                f"{case}{suffix}"
                for case in cases
                for suffix in (".csv", ".pdf", "-paper.pdf", "-paper-global.pdf")
            },
        )
        self.assertEqual(len(list(output.glob("*.pdf"))), 27)
        for case in cases:
            with (output / f"{case}.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            selected = {
                (row["degree"], row["block_size"], row["items_per_thread"])
                for row in rows
                if row["backend"] == "bits-sq"
                and row["paper_global_selected"] == "True"
            }
            self.assertEqual(
                selected,
                {("8", "128", "4")} if case.endswith("small") else {("32", "512", "8")},
            )
            self.assertTrue(all("scaling_global_selected" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
