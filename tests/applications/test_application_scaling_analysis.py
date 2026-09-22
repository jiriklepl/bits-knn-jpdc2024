"""Exercise complete scaling studies, provenance checks, and rendered comparisons."""

import copy
import csv
from decimal import Decimal, ROUND_HALF_UP
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

    def make_study(self, *, tiers=("small", "large"), retention=True):
        settings = dict(
            ks=[32, 64],
            degrees=[8, 32],
            blocks=[128, 512],
            items=[4, 8],
            repeat=3,
            warmup=1,
            retention_ratios=[0.01, 0.02] if retention else [],
        )
        index = dict(version=1, settings=settings, workloads=[], runs=[])
        for operator in analysis.SIZE_LABELS:
            for position, suffix in enumerate(tiers):
                identity = f"{operator}-{suffix}"
                sizes = (512, 1024, 2048) if len(tiers) == 3 else (512, 2048)
                batches = (2, 4, 8) if len(tiers) == 3 else (2, 8)
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
                modes = [("fixed-k", None, settings["ks"])]
                if operator == "gradient-compression":
                    modes.extend(
                        (
                            "retention",
                            ratio,
                            [
                                int(
                                    (
                                        Decimal(str(ratio)) * candidates
                                    ).to_integral_value(rounding=ROUND_HALF_UP)
                                )
                            ],
                        )
                        for ratio in settings["retention_ratios"]
                    )
                for mode, ratio, ks in modes:
                    run = dict(
                        workload_id=identity,
                        mode=mode,
                        requested_retention=ratio,
                        ks=ks,
                        csv=(
                            f"{identity}.csv"
                            if not retention
                            else f"{identity}-{mode}-{ratio}.csv"
                        ),
                        status="complete",
                    )
                    if ratio is not None:
                        run["actual_retention"] = ks[0] / candidates
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
        self.index = self.make_study(
            tiers=("small", "middle", "large"), retention=False
        )
        self.save_index()
        cli = self.cli()
        output = self.root / "plots"
        args = [cli.__file__, str(self.path), "--output-dir", str(output)]
        with patch.object(sys, "argv", args), patch.object(
            sys, "stderr", io.StringIO()
        ), patch.object(
            cli, "write_outputs"
        ) as write, patch.object(cli.subprocess, "run") as execute, patch.object(
            analysis, "annotate_choices"
        ) as select_across_sizes:
            cli.main()
        write.assert_not_called()
        select_across_sizes.assert_not_called()
        self.assertEqual(execute.call_count, 9)
        commands = [call.args[0] for call in execute.call_args_list]
        cases = {
            f"{operator}-{tier}"
            for operator in analysis.SIZE_LABELS
            for tier in ("small", "middle", "large")
        }
        self.assertEqual({Path(command[2]).stem for command in commands}, cases)
        for command in commands:
            operator = next(
                name
                for name in analysis.SIZE_LABELS
                if Path(command[2]).stem.startswith(name)
            )
            self.assertEqual(
                command[:2], [sys.executable, str(SCRIPTS / f"plot-{operator}.py")]
            )
            self.assertEqual(command[-2:], ["--output-dir", str(output)])
        self.assertTrue(all(call.kwargs["check"] for call in execute.call_args_list))

    def test_default_rejects_legacy_study_before_outputs(self):
        self.reject_before_outputs("exactly nine fixed-k cases.*--cross-size")

    def test_default_requires_every_size_tier_for_each_application(self):
        self.index = self.make_study(
            tiers=("small", "middle", "large"), retention=False
        )
        original = copy.deepcopy(self.index)
        for failure in ("missing_tier", "duplicate_tier", "unknown_tier", "filename"):
            with self.subTest(failure=failure):
                self.index = copy.deepcopy(original)
                workload = self.index["workloads"][1]
                if failure == "missing_tier":
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

    def test_skip_individual_runs_requires_explicit_cross_size_mode(self):
        cli = self.cli()
        with patch.object(
            sys, "argv", [cli.__file__, str(self.path), "--skip-individual-runs"]
        ), patch.object(sys, "stderr", io.StringIO()) as stderr, patch.object(
            cli, "load_study"
        ) as load, self.assertRaises(
            SystemExit
        ) as error:
            cli.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn(
            "--skip-individual-runs requires --cross-size", stderr.getvalue()
        )
        load.assert_not_called()

    def test_cross_size_mode_preserves_legacy_outputs_and_individual_opt_out(self):
        cli = self.cli()
        output = self.root / "plots"
        for skip in (False, True):
            args = [
                cli.__file__,
                str(self.path),
                "--output-dir",
                str(output),
                "--cross-size",
            ]
            if skip:
                args.append("--skip-individual-runs")
            with self.subTest(skip=skip), patch.object(sys, "argv", args), patch.object(
                sys, "stderr", io.StringIO()
            ), patch.object(
                cli, "write_outputs"
            ) as write, patch.object(cli.subprocess, "run") as execute:
                cli.main()
            write.assert_called_once()
            self.assertEqual(execute.call_count, 0 if skip else 10)
            if not skip:
                commands = [call.args[0] for call in execute.call_args_list]
                self.assertEqual(
                    {Path(command[1]).name for command in commands},
                    {
                        "plot-database-topn.py",
                        "plot-token-sampling.py",
                        "plot-gradient-compression.py",
                    },
                )
                self.assertEqual(
                    {Path(command[2]).name for command in commands},
                    {run["csv"] for run in self.index["runs"]},
                )
                self.assertTrue(
                    all(
                        command[-2:] == ["--output-dir", str(output / "workloads")]
                        for command in commands
                    )
                )
                self.assertTrue(
                    all(call.kwargs["check"] for call in execute.call_args_list)
                )

    def test_global_choices_do_not_pool_operators_or_retention_modes(self):
        rows = analysis.load_study(self.path)
        for row in rows:
            if (
                row["mode"] != "retention"
                or row["backend"] != "bits-sq"
                or row["phase"] != "operator"
            ):
                continue
            config = row["degree"], row["block_size"], row["items_per_thread"]
            row["median_ms"] = (
                (1 if row["scenario"] == 0.01 else 100)
                if config == (8, 128, 4)
                else 4
                if config == (32, 512, 8)
                else 200
            )
        selected = analysis.annotate_choices(rows, self.path)
        for row in selected:
            if row["backend"] != "bits-sq":
                continue
            config = row["degree"], row["block_size"], row["items_per_thread"]
            expected = (32, 512, 8) if row["mode"] == "retention" else (8, 128, 4)
            self.assertEqual(row["scaling_global_selected"], config == expected)
            if row["mode"] == "retention":
                point = (8, 128, 4) if row["scenario"] == 0.01 else (32, 512, 8)
                self.assertEqual(row["scaling_selected"], config == point)

    def test_verified_study_groups_sizes_and_ratios_without_pooling_or_mixing_choices(
        self
    ):
        rows = analysis.load_study(self.path)
        self.assertEqual(len(self.index["runs"]), 10)
        self.assertEqual({row["mode"] for row in rows}, {"fixed-k", "retention"})
        for row in rows:
            backend = row["backend"]
            config = row["degree"], row["block_size"], row["items_per_thread"]
            small = row["workload_id"].endswith("small")
            if backend == "bits-sq":
                self.assertEqual(
                    row["scaling_selected"],
                    config == ((8, 128, 4) if small else (32, 512, 8)),
                )
                self.assertEqual(row["scaling_global_selected"], config == (8, 128, 4))
                self.assertEqual(
                    row["degree_selected"], config in ((8, 128, 4), (32, 512, 8))
                )
            elif backend == "bits-prefetch":
                self.assertEqual(
                    row["scaling_selected"],
                    config == ((1, 128, 4) if small else (1, 512, 8)),
                )
                self.assertEqual(row["scaling_global_selected"], config == (1, 128, 4))
            else:
                for flag in (
                    "scaling_selected",
                    "scaling_global_selected",
                    "degree_selected",
                ):
                    self.assertEqual(row[flag], backend != "block-select")
            if row["mode"] == "retention":
                self.assertEqual(row["scenario"], row["requested_retention"])
                self.assertAlmostEqual(row["retention_ratio"], row["k"] / row["rows"])
            self.assertAlmostEqual(
                row["throughput_mvalues_per_s"],
                row["rows"] * row.get("batch_size", 1) / (1000 * row["median_ms"]),
            )
        # One global choice spans both workload sizes and both requested ratios;
        # its isolated-selection measurements follow the operator choice.
        selected = [
            row
            for row in rows
            if row["mode"] == "retention"
            and row["backend"] == "bits-sq"
            and row["scaling_global_selected"]
        ]
        self.assertEqual({row["k"] for row in selected}, {5, 10, 20, 41})
        self.assertEqual(
            {row["phase"] for row in selected},
            {
                "upload_shared",
                "operator",
                "download",
                "selection_isolated",
                "transform_isolated",
                "output_isolated",
            },
        )
        self.assertEqual(
            {
                round(row["median_ms"], 5)
                for row in selected
                if row["phase"] == "selection_isolated"
            },
            {20, round(20 / 9, 5)},
        )

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
            ("missing_ratio", "every workload"),
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
                elif failure in ("missing_run", "missing_ratio"):
                    self.index["runs"].pop(0 if failure == "missing_run" else -1)
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

    def test_missing_configuration_is_not_accepted_as_a_partial_global_candidate(self):
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

    def test_retention_scenario_must_match_rounded_requested_ratio(self):
        run = next(item for item in self.index["runs"] if item["mode"] == "retention")
        run["actual_retention"] *= 2
        self.save_index()
        self.reject_before_outputs("Actual retention")


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
        self.index = self.make_study(
            tiers=("small", "middle", "large"), retention=False
        )
        self.save_index()
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
            for operator in analysis.SIZE_LABELS
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

    def test_global_configurations_degree_curves_and_legend_bounds(self):
        with patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / "cache")}):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib.container import ErrorbarContainer
            import numpy as np

            self.addCleanup(plt.close, "all")
            captured = []

            class CapturePdf:
                def __init__(self, filename):
                    self.filename = Path(filename)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def savefig(self, figure, **_kwargs):
                    figure.canvas.draw()
                    captured.append((self.filename, figure))

            rows = analysis.load_study(self.path)
            output = self.root / "plots"
            with patch("matplotlib.backends.backend_pdf.PdfPages", CapturePdf):
                analysis.write_outputs(rows, output)
            self.assertEqual(len(captured), 24)
            self.assertEqual(len({path.name for path, _ in captured}), 12)
            for path, figure in captured:
                global_plot = path.stem.endswith("-paper-global")
                degree_plot = path.stem.endswith("-degree")
                legend = (
                    figure.axes[0].get_legend() if degree_plot else figure.legends[0]
                )
                labels = [text.get_text() for text in legend.get_texts()]
                self.assertFalse(any("BlockSelect" in label for label in labels))
                if global_plot:
                    self.assertEqual(
                        labels[:2],
                        [
                            "bits [block=128, items=4]",
                            "bits (split) [degree=8, block=128, items=4]",
                        ],
                    )
                renderer = figure.canvas.get_renderer()
                bounds = legend.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, figure.bbox.x0)
                self.assertGreaterEqual(bounds.y0, figure.bbox.y0)
                self.assertLessEqual(bounds.x1, figure.bbox.x1)
                self.assertLessEqual(bounds.y1, figure.bbox.y1)
                if not degree_plot:
                    self.assertLessEqual(
                        bounds.y1, min(ax.bbox.y0 for ax in figure.axes)
                    )
                    for axis in figure.axes:
                        self.assertEqual(axis.get_title(), "")
                if global_plot:
                    curves = [
                        item
                        for item in figure.axes[0].containers
                        if isinstance(item, ErrorbarContainer)
                    ]
                    self.assertEqual(len(curves), 4)
                    np.testing.assert_allclose(
                        curves[0].lines[0].get_ydata(orig=False), [2, 8]
                    )
                    np.testing.assert_allclose(
                        curves[1].lines[0].get_ydata(orig=False), [1, 9]
                    )
                if degree_plot:
                    curves = [
                        item
                        for item in figure.axes[0].containers
                        if isinstance(item, ErrorbarContainer)
                    ]
                    self.assertEqual(len(curves), 2)
                    for curve in curves:
                        np.testing.assert_equal(
                            curve.lines[0].get_xdata(orig=False), [8, 32]
                        )
                    np.testing.assert_allclose(
                        curves[0].lines[0].get_ydata(orig=False), [20, 5]
                    )
                    np.testing.assert_allclose(
                        curves[1].lines[0].get_ydata(orig=False), [20 / 9, 5]
                    )
            with (output / "application-scaling.csv").open(newline="") as source:
                written = list(csv.DictReader(source))
            self.assertEqual(len(written), len(rows))
            self.assertTrue(all("scaling_global_selected" in row for row in written))


if __name__ == "__main__":
    unittest.main()
