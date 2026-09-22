"""Validate scaling plans and checkpoint behavior with a CPU native-runner stand-in."""

from collections import Counter
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_inputs import GRADIENT_SEMANTICS, sha256_file  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "scaling_runner", SCRIPTS / "run-application-scaling.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FAKE_NATIVE = """#!/usr/bin/env python3
import csv, json, os, sys
options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
with open(os.environ["SCALING_CALLS"], "a") as output:
    output.write(json.dumps(options) + "\\n")
with open(os.environ["SCALING_CALLS"]) as calls:
    call_count = sum(1 for _ in calls)
if str(call_count) == os.environ.get("SCALING_FAIL_CALL"):
    print("simulated verification failure", file=sys.stderr)
    sys.exit(7)
writer = csv.writer(sys.stdout)
writer.writerow(["operator", "dataset_id", "backend", "rows", "batch_size", "k",
                 "retention_ratio", "degree", "block_size", "items_per_thread",
                 "iteration", "phase", "seconds"])
for backend in options["--backends"].split(","):
    for phase in ("upload_shared", "operator", "download", "transform_isolated",
                  "selection_isolated", "output_isolated"):
        iterations = ([-1] if phase == "upload_shared"
                      else range(int(options["--repeat"])))
        for iteration in iterations:
            writer.writerow(["gradient-compression", options["--dataset-id"], backend,
                             options["--elements"], 1, options["-k"],
                             int(options["-k"])/int(options["--elements"]),
                             options["--degree"] if backend == "bits-sq" else 1,
                             options["--bits-block-size"],
                             options["--items-per-thread"],
                             iteration, phase, 0.0001])
"""


class ScalingPlanTests(unittest.TestCase):
    def test_default_cli_plans_nine_named_cases(self):
        with mock.patch.object(
            sys,
            "argv",
            ["run-application-scaling.py", "suite.json", "--output-dir", "results"],
        ), mock.patch.object(runner, "execute", return_value=0) as execute:
            self.assertEqual(runner.main(), 0)
        args = execute.call_args.args[0]
        self.assertEqual(args.warmup, 10)
        self.assertEqual(args.repeat, 30)
        workloads, manifests = [], {}
        for operator, field, sizes in (
            ("database-topn", "rows", (600572, 6001215, 59986052)),
            ("token-sampling", "vocabulary_size", (50257, 50257, 50257)),
            ("gradient-compression", "elements", (589824, 2359296, 38597376)),
        ):
            for tier, size in zip(("small", "middle", "large"), sizes):
                identity = f"input-{operator}-{tier}"
                workloads.append(dict(id=identity, operator=operator, size_tier=tier))
                manifests[identity] = (None, {field: size})
        records = runner.make_runs(workloads, manifests, args.ks, args.degrees)
        self.assertEqual(len(records), 9)
        self.assertEqual({record["mode"] for record in records}, {"fixed-k"})
        self.assertEqual({record["status"] for record in records}, {"pending"})
        self.assertEqual(
            {record["csv"] for record in records},
            {
                f"{operator}-{tier}.csv"
                for operator in runner.OPERATORS
                for tier in ("small", "middle", "large")
            },
        )
        self.assertTrue(
            all(
                record["log"] == record["csv"].removesuffix(".csv") + ".err"
                for record in records
            )
        )
        self.assertEqual(
            sum(
                len(config["backends"])
                for record in records
                for config in runner.configurations(record["ks"], args.degrees)
            ),
            4212,
        )

    def test_tiered_output_names_cannot_collide(self):
        workload = dict(
            id="attention", operator="gradient-compression", size_tier="small"
        )
        manifests = {"attention": (None, {"elements": 589824})}
        records = runner.make_runs([workload], manifests, [32], [32])
        self.assertEqual(
            [record["csv"] for record in records],
            ["gradient-compression-small.csv"],
        )
        other = workload | {"id": "another-attention"}
        manifests[other["id"]] = manifests[workload["id"]]
        with self.assertRaisesRegex(ValueError, "output filenames collide"):
            runner.make_runs([workload, other], manifests, [32], [32])

    def test_compute_idle_check_blocks_other_processes_on_benchmark_gpu(self):
        completed = subprocess.CompletedProcess(
            [], 0, "GPU-test, 123, another-benchmark\n"
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}):
                with self.assertRaisesRegex(ValueError, "PID 123"):
                    runner.ensure_gpu_idle([{"uuid": "GPU-test"}])
                runner.ensure_gpu_idle([{"uuid": "GPU-other"}])

    def test_complete_configuration_matrix_deduplicates_plain_bits_and_baselines(self):
        configs = list(runner.configurations([32, 59, 2048], [8, 32, 128, 512]))
        counts = Counter(
            (config["k"], backend)
            for config in configs
            for backend in config["backends"]
        )
        for k in (32, 59, 2048):
            self.assertEqual(counts[k, "bits-prefetch"], 15)
            self.assertEqual(counts[k, "bits-sq"], 60)
            self.assertEqual(counts[k, "air-topk"], 1)
            self.assertEqual(counts[k, "grid-select"], 1)
        self.assertEqual(counts[32, "block-select"], 1)
        self.assertEqual(counts[59, "block-select"], 0)
        self.assertEqual(counts[2048, "block-select"], 0)

    def test_untiered_names_and_unsupported_values_are_explicit(self):
        workloads = [dict(id="gradient", operator="gradient-compression")]
        manifests = {"gradient": (None, {"elements": 2359296})}
        records = runner.make_runs(workloads, manifests, [32, 59], [32])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["csv"], "gradient-fixed-k.csv")
        self.assertEqual(records[0]["ks"], [32, 59])
        self.assertEqual(records[0]["status"], "pending")
        for elements, ks, degrees, reason in (
            (2359296, [2049], [32], "k=2049"),
            (16, [32], [8], "candidates=16"),
            (64, [32], [128], "split degree 128 exceeds 64"),
        ):
            with self.subTest(elements=elements, ks=ks, degrees=degrees):
                manifests["gradient"] = (None, {"elements": elements})
                record = runner.make_runs(workloads, manifests, ks, degrees)[0]
                self.assertEqual(record["status"], "unsupported")
                self.assertIn(reason, record["reason"])

    def test_partial_and_duplicate_native_csv_are_rejected(self):
        configuration = dict(
            k=32, block=128, items=4, degree=8, backends=["bits-prefetch"]
        )
        header = (
            "dataset_id,backend,k,degree,block_size,items_per_thread,"
            "iteration,phase,seconds\n"
        )
        row = "abc,bits-prefetch,32,1,128,4,0,operator,0.1\n"
        for data in (header, header + row, header + row + row):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                runner.validate_chunk(data, configuration, "abc", 1)


class ScalingRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        gradient = self.root / "gradient.f32"
        gradient.write_bytes(bytes(4096 * 4))
        manifest = dict(
            version=1,
            operator="gradient-compression",
            elements=4096,
            tensor_shape=[4096],
            semantics=GRADIENT_SEMANTICS,
            source={"kind": "test"},
            columns={
                "gradient": dict(
                    file="gradient.f32",
                    dtype="float32",
                    byte_order="little",
                    shape=[4096],
                    sha256=sha256_file(gradient),
                )
            },
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        workload = dict(
            id="gradient",
            operator="gradient-compression",
            label="Test gradient",
            manifest="manifest.json",
            dataset_id=sha256_file(self.manifest),
            size=4096,
            size_unit="elements",
            score_bytes=4096 * 4,
        )
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(dict(version=1, workloads=[workload])))
        self.binary = self.root / "gradient-compression"
        self.binary.write_text(FAKE_NATIVE)
        self.binary.chmod(0o755)
        self.calls = self.root / "calls.jsonl"
        self.args = type(
            "Args",
            (),
            dict(
                suite=self.suite,
                build_dir=self.root,
                output_dir=self.root / "results",
                ks=[32, 64],
                degrees=[8, 32],
                warmup=0,
                repeat=2,
                resume=False,
                dry_run=False,
            ),
        )()
        for patch in (
            mock.patch.object(
                runner,
                "gpu_identity",
                return_value=[
                    {
                        "uuid": "test-gpu",
                        "name": "CPU stand-in",
                        "driver_version": "test",
                    }
                ],
            ),
            mock.patch.object(runner, "BLOCKS", (128,)),
            mock.patch.object(runner, "ITEMS", (4,)),
            mock.patch.object(runner, "ensure_gpu_idle"),
            mock.patch.dict(os.environ, {"SCALING_CALLS": str(self.calls)}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def execute(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return runner.execute(self.args)

    def index(self):
        return json.loads((self.args.output_dir / "index.json").read_text())

    def call_count(self):
        return len(self.calls.read_text().splitlines())

    def test_atomic_failure_checkpoint_resume_and_hash_guards(self):
        suite = json.loads(self.suite.read_text())
        workload = suite["workloads"][0]
        suite["workloads"] += [
            workload | {"id": "gradient-second"},
            workload | {"id": "gradient-third"},
        ]
        self.suite.write_text(json.dumps(suite))
        with mock.patch.dict(os.environ, {"SCALING_FAIL_CALL": "5"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute()
        index = self.index()
        self.assertEqual(
            [record["status"] for record in index["runs"]],
            ["complete", "failed", "pending"],
        )
        first = self.args.output_dir / index["runs"][0]["csv"]
        first_hash = sha256_file(first)
        self.assertFalse((self.args.output_dir / index["runs"][1]["csv"]).exists())
        self.assertFalse(list(self.args.output_dir.glob("*.partial")))
        self.assertEqual(self.call_count(), 5)
        self.args.resume = True
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.call_count(), 13)
        self.assertEqual(sha256_file(first), first_hash)
        index = self.index()
        self.assertTrue(all(record["status"] == "complete" for record in index["runs"]))
        self.assertEqual(index["settings"]["gpu"][0]["uuid"], "test-gpu")
        for record in index["runs"]:
            self.assertEqual(
                record["csv_sha256"], sha256_file(self.args.output_dir / record["csv"])
            )
        self.execute()
        self.assertEqual(self.call_count(), 13)
        with mock.patch.object(
            runner, "gpu_identity", return_value=[{"uuid": "another-gpu"}]
        ):
            with self.assertRaisesRegex(ValueError, "settings_sha256 differs"):
                self.execute()
        first.write_text(first.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "timing hash differs"):
            self.execute()

    def test_invalid_input_and_settings_fail_before_native_execution(self):
        self.args.degrees = [8, 8]
        with self.assertRaisesRegex(ValueError, "Duplicate degrees"):
            self.execute()
        self.args.degrees = [8]
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Manifest SHA-256 mismatch"):
            self.execute()
        self.assertFalse(self.calls.exists())

    def test_size_tiers_are_validated_and_partial_suites_remain_supported(self):
        suite = json.loads(self.suite.read_text())
        workload = suite["workloads"][0]
        for tier in ("small", "middle", "large"):
            with self.subTest(tier=tier):
                workload["size_tier"] = tier
                self.suite.write_text(json.dumps(suite))
                loaded, _ = runner.load_suite(self.suite)
                self.assertEqual(loaded["workloads"][0]["size_tier"], tier)
        for tier in ("medium", "", None, [], True):
            with self.subTest(invalid=tier):
                workload["size_tier"] = tier
                self.suite.write_text(json.dumps(suite))
                with self.assertRaisesRegex(ValueError, "Invalid size tier"):
                    runner.load_suite(self.suite)
        workload["size_tier"] = "small"
        suite["workloads"].append(workload | {"id": "another-gradient"})
        self.suite.write_text(json.dumps(suite))
        with self.assertRaisesRegex(ValueError, "Duplicate size tier"):
            runner.load_suite(self.suite)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
