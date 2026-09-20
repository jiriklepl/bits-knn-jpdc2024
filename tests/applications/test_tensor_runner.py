"""Opt in with TOKEN_SAMPLING_BINARY and/or GRADIENT_COMPRESSION_BINARY."""

import csv
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_inputs import (  # noqa: E402
    GRADIENT_SEMANTICS,
    SAMPLING_SEMANTICS,
    load_application_manifest,
    sha256_file,
)

BACKENDS = {"bits-prefetch", "bits-sq", "air-topk", "grid-select", "block-select"}
BACKEND_LIST = "bits-prefetch,bits-sq,air-topk,grid-select,block-select"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class TensorFixture:
    """Synthetic tensors exercise integration, without posing as model captures."""

    def __init__(self, root, operator):
        self.root = root
        self.operator = operator
        self.sampling = operator == "token-sampling"
        self.batch = 3 if self.sampling else 1
        self.rows = 131 if self.sampling else 141
        self.column = "logits" if self.sampling else "gradient"
        if self.sampling:
            self.values = [
                f32(((j * 7 + row * 11) % 19 - 9) / 4) if j < 110 else -math.inf
                for row in range(self.batch)
                for j in range(self.rows)
            ]
            shape = [self.batch, self.rows]
            dimensions = {"batch_size": self.batch, "vocabulary_size": self.rows}
        else:
            self.values = [f32((j % 23 - 11) * 0.125) for j in range(self.rows)]
            self.values[0] = -0.0
            shape = [3, 47]
            dimensions = {"elements": self.rows, "tensor_shape": shape}
        self.raw = root / (self.column + ".f32")
        self.raw.write_bytes(struct.pack(f"<{len(self.values)}f", *self.values))
        self.manifest = root / "manifest.json"
        self.metadata = {
            "version": 1,
            "operator": operator,
            **dimensions,
            "semantics": SAMPLING_SEMANTICS if self.sampling else GRADIENT_SEMANTICS,
            "source": {"kind": "synthetic-test"},
            "columns": {
                self.column: {
                    "file": self.raw.name,
                    "dtype": "float32",
                    "shape": shape,
                    "byte_order": "little",
                    "sha256": sha256_file(self.raw),
                }
            },
        }
        self.manifest.write_text(json.dumps(self.metadata), encoding="utf-8")
        load_application_manifest(self.manifest)

    def native_command(self, binary):
        command = [str(binary), "--" + self.column, str(self.raw)]
        if self.sampling:
            command += [
                "--batch-size",
                str(self.batch),
                "--vocabulary-size",
                str(self.rows),
            ]
        else:
            command += ["--elements", str(self.rows)]
        return command

    def launcher_command(self, binary):
        return [
            sys.executable,
            str(SCRIPTS / "run-applications.py"),
            str(self.manifest),
            "--binary",
            str(binary),
        ]


class TensorNativeChecks:
    """Shared checks inherited only by the independently enabled GPU test classes."""

    def setUp(self):
        self.binary = Path(os.environ[self.binary_variable]).resolve()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = TensorFixture(self.root, self.operator)

    def run_native(self, options=(), overrides=None):
        command = self.fixture.native_command(self.binary)
        for name, value in (overrides or {}).items():
            command[command.index(name) + 1] = str(value)
        if "--warmup" not in options:
            command += ["--warmup", "0"]
        if "--repeat" not in options:
            command += ["--repeat", "1"]
        return subprocess.run(command + list(options), text=True, capture_output=True)

    def test_manifest_comparison_schema_configuration_and_output(self):
        output = self.root / "result.csv"
        seed = 2**64 - 1 if self.fixture.sampling else 0
        temperature = f32(0.7) if self.fixture.sampling else 0.0
        command = self.fixture.launcher_command(self.binary) + [
            "--k",
            "32",
            "--backends",
            BACKEND_LIST,
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output",
            str(output),
        ]
        if self.fixture.sampling:
            command += ["--temperature", "0.7", "--seed", str(seed)]
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        provenance = json.loads(run.stderr.splitlines()[0])
        self.assertEqual(provenance["operator"], self.operator)
        self.assertEqual(provenance["dataset_id"], sha256_file(self.fixture.manifest))
        self.assertEqual(provenance["binary_sha256"], sha256_file(self.binary))
        if self.fixture.sampling:
            args = provenance["command"]
            self.assertEqual(args[args.index("--seed") + 1], str(seed))
        timings = list(csv.DictReader(io.StringIO(run.stdout)))
        self.assertEqual(len(timings), 5 * (1 + 2 * 5))
        self.assertEqual({r["backend"] for r in timings}, BACKENDS)
        phases = {
            "operator",
            "download",
            "upload_shared",
            "transform_isolated",
            "selection_isolated",
            "output_isolated",
        }
        self.assertEqual({r["phase"] for r in timings}, phases)
        for backend in BACKENDS:
            selected = [r for r in timings if r["backend"] == backend]
            for phase in phases:
                iterations = [
                    int(r["iteration"]) for r in selected if r["phase"] == phase
                ]
                self.assertEqual(
                    iterations, [-1] if phase == "upload_shared" else [0, 1]
                )
        for row in timings:
            self.assertEqual(row["operator"], self.operator)
            self.assertEqual(row["dataset_id"], sha256_file(self.fixture.manifest))
            self.assertEqual(int(row["rows"]), self.fixture.rows)
            self.assertEqual(int(row["batch_size"]), self.fixture.batch)
            self.assertEqual(int(row["k"]), 32)
            self.assertEqual(row["seed"], str(seed))
            self.assertEqual(float(row["temperature"]), temperature)
            self.assertTrue(math.isfinite(float(row["seconds"])))
            self.assertGreaterEqual(float(row["seconds"]), 0)
            self.assertAlmostEqual(
                float(row["retention_ratio"]), 32 / self.fixture.rows
            )
            expected = {
                "bits-prefetch": (1, 512, 7),
                "bits-sq": (32, 512, 4),
                "air-topk": (1, 512, 0),
                "grid-select": (1, 0, 0),
                "block-select": (1, 128, 2),
            }[row["backend"]]
            self.assertEqual(
                tuple(
                    int(row[key])
                    for key in ("degree", "block_size", "items_per_thread")
                ),
                expected,
            )
        with output.open() as source:
            results = list(csv.DictReader(source))
        self.check_output(results, BACKENDS, 32, temperature)

    def check_output(self, results, backends, k, temperature):
        self.assertEqual({r["backend"] for r in results}, backends)
        self.assertEqual(
            len(results),
            len(backends) * (self.fixture.batch if self.fixture.sampling else k),
        )
        for backend in backends:
            selected = [r for r in results if r["backend"] == backend]
            if self.fixture.sampling:
                self.assertEqual(
                    [int(r["sequence"]) for r in selected],
                    list(range(self.fixture.batch)),
                )
                for row in selected:
                    sequence = int(row["sequence"])
                    token = int(row["token"])
                    self.assertTrue(0 <= token < self.fixture.rows)
                    values = self.fixture.values[
                        sequence * self.fixture.rows : (sequence + 1)
                        * self.fixture.rows
                    ]
                    support = sorted(values, reverse=True)[:k]
                    self.assertTrue(math.isfinite(values[token]))
                    self.assertGreaterEqual(values[token], support[-1])
                    weights = [
                        math.exp((value - support[0]) / temperature)
                        for value in support
                    ]
                    expected = math.exp(
                        (values[token] - support[0]) / temperature
                    ) / sum(weights)
                    actual = float(row["probability"])
                    self.assertGreater(actual, 0)
                    self.assertLessEqual(actual, 1)
                    self.assertAlmostEqual(actual, expected, delta=2e-6 * expected)
            else:
                indices = [int(r["index"]) for r in selected]
                self.assertEqual(len(set(indices)), k)
                for row, index in zip(selected, indices):
                    self.assertTrue(0 <= index < self.fixture.rows)
                    self.assertEqual(
                        struct.pack("<f", float(row["value"])),
                        struct.pack("<f", self.fixture.values[index]),
                    )
                self.assertEqual(
                    sorted(
                        (abs(float(row["value"])) for row in selected), reverse=True
                    ),
                    sorted(map(abs, self.fixture.values), reverse=True)[:k],
                )

    def test_arbitrary_k_without_block_select(self):
        output = self.root / "arbitrary.csv"
        backends = BACKENDS - {"block-select"}
        run = self.run_native(
            [
                "-k",
                "7",
                "--backends",
                ",".join(sorted(backends)),
                "--output",
                str(output),
            ]
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        with output.open() as source:
            self.check_output(list(csv.DictReader(source)), backends, 7, 1.0)

    def test_default_backends_and_explicit_bits_configuration(self):
        run = self.run_native()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(
            {r["backend"] for r in csv.DictReader(io.StringIO(run.stdout))},
            BACKENDS - {"bits-prefetch"},
        )
        for block, batch in [(128, 1), (256, 4), (512, 16), (128, 8), (256, 13)]:
            with self.subTest(block=block, batch=batch):
                run = self.run_native(
                    [
                        "--backends",
                        "bits-prefetch,bits-sq",
                        "--degree",
                        "7",
                        "--bits-block-size",
                        str(block),
                        "--items-per-thread",
                        str(batch),
                    ]
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                for row in csv.DictReader(io.StringIO(run.stdout)):
                    self.assertEqual(int(row["block_size"]), block)
                    self.assertEqual(int(row["items_per_thread"]), batch)
                    self.assertEqual(
                        int(row["degree"]), 7 if row["backend"] == "bits-sq" else 1
                    )

    def test_invalid_configurations_fail(self):
        cases = [
            ["-k", "0"],
            ["-k", str(self.fixture.rows + 1)],
            ["-k", "7"],
            ["-k", "32junk"],
            ["--degree", "0"],
            ["--degree", str(self.fixture.rows + 1)],
            ["--bits-block-size", "129"],
            ["--items-per-thread", "2"],
            ["--repeat", "0"],
            ["--backends", "unknown"],
            ["--backends", "bits-prefetch,bits-prefetch"],
            ["--backends", "bits-prefetch,"],
            ["--backends", ""],
            ["--dataset-id", "bad"],
        ]
        for options in cases:
            with self.subTest(options=options):
                run = self.run_native(options)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr.strip())

    def test_wrong_raw_shapes_bytes_and_nonfinite_values_fail(self):
        original = self.fixture.raw.read_bytes()
        bad_values = (
            [math.nan, math.inf]
            if self.fixture.sampling
            else [math.nan, math.inf, -math.inf]
        )
        contents = [original[:-1], original + b"\0"]
        contents += [struct.pack("<f", value) + original[4:] for value in bad_values]
        if self.fixture.sampling:
            # A whole masked sequence fails even if the other sequences remain valid.
            contents += [
                struct.pack(
                    f"<{self.fixture.rows}f", *([-math.inf] * self.fixture.rows)
                )
                + original[self.fixture.rows * 4 :]
            ]
        for content in contents:
            with self.subTest(size=len(content), prefix=content[:4]):
                self.fixture.raw.write_bytes(content)
                run = self.run_native()
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr.strip())
        self.fixture.raw.write_bytes(original)
        if self.fixture.sampling:
            shapes = [
                {"--batch-size": 0},
                {"--batch-size": 4},
                {"--vocabulary-size": 0},
                {"--vocabulary-size": 132},
                {"--batch-size": 2**31},
            ]
        else:
            shapes = [{"--elements": 0}, {"--elements": 142}, {"--elements": 2**31}]
        for overrides in shapes:
            with self.subTest(shape=overrides):
                self.assertNotEqual(self.run_native(overrides=overrides).returncode, 0)

    def test_existing_output_is_preserved(self):
        output = self.root / "keep.csv"
        original = b"existing experiment\n"
        output.write_bytes(original)
        run = self.run_native(["--output", str(output)])
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        run = subprocess.run(
            self.fixture.launcher_command(self.binary) + ["--output", str(output)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_deferred_output_failure_does_not_publish_timings(self):
        # This fails only when writing the optional results, after GPU execution
        # and both verification passes. Partial timing CSV must remain unpublished.
        output = self.root / "missing-parent" / "result.csv"
        run = self.run_native(["--backends", "bits-prefetch", "--output", str(output)])
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Writing output failed", run.stderr)
        self.assertEqual(run.stdout, "")
        self.assertFalse(output.exists())


@unittest.skipUnless(
    os.environ.get("TOKEN_SAMPLING_BINARY"),
    "Set TOKEN_SAMPLING_BINARY for GPU integration tests",
)
class SamplingRunnerTests(TensorNativeChecks, unittest.TestCase):
    binary_variable = "TOKEN_SAMPLING_BINARY"
    operator = "token-sampling"

    def test_temperature_and_seed_validation(self):
        for temperature in ["junk", "1junk", "0", "-1", "nan", "inf", "1e-60", "1e40"]:
            with self.subTest(temperature=temperature):
                self.assertNotEqual(
                    self.run_native(["--temperature", temperature]).returncode, 0
                )
        for seed in ["-1", "1junk", str(2**64)]:
            with self.subTest(seed=seed):
                self.assertNotEqual(self.run_native(["--seed", seed]).returncode, 0)
        for launcher in [False, True]:
            with self.subTest(launcher=launcher):
                options = ["--temperature", "1e-45"]
                if launcher:
                    run = subprocess.run(
                        self.fixture.launcher_command(self.binary)
                        + options
                        + ["--warmup", "0", "--repeat", "1"],
                        text=True,
                        capture_output=True,
                    )
                else:
                    run = self.run_native(options)
                self.assertEqual(run.returncode, 0, run.stderr)
                for row in csv.DictReader(io.StringIO(run.stdout)):
                    self.assertEqual(float(row["temperature"]), f32(1e-45))


@unittest.skipUnless(
    os.environ.get("GRADIENT_COMPRESSION_BINARY"),
    "Set GRADIENT_COMPRESSION_BINARY for GPU integration tests",
)
class GradientRunnerTests(TensorNativeChecks, unittest.TestCase):
    binary_variable = "GRADIENT_COMPRESSION_BINARY"
    operator = "gradient-compression"

    def test_sampling_only_flags_rejected(self):
        for options in [["--temperature", "1"], ["--seed", "42"]]:
            with self.subTest(options=options):
                self.assertNotEqual(self.run_native(options).returncode, 0)
                run = subprocess.run(
                    self.fixture.launcher_command(self.binary) + options,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("only apply to token-sampling", run.stderr)


class TensorLauncherTests(unittest.TestCase):
    """Launcher behavior is checked without requiring a GPU or model dependencies."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_failure_propagation_and_exact_sampling_seed(self):
        for operator in ["token-sampling", "gradient-compression"]:
            with self.subTest(operator=operator):
                root = self.root / operator
                root.mkdir()
                fixture = TensorFixture(root, operator)
                arguments = root / "arguments.json"
                binary = root / "fake binary"
                binary.write_text(
                    f"#!{sys.executable}\nimport json, sys\n"
                    f"open({str(arguments)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                    "sys.exit(7)\n"
                )
                binary.chmod(0o755)
                command = fixture.launcher_command(binary)
                if fixture.sampling:
                    command += ["--seed", str(2**64 - 1), "--temperature", "0.7"]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 7)
                args = json.loads(arguments.read_text())
                self.assertEqual(
                    args[args.index("--" + fixture.column) + 1], str(fixture.raw)
                )
                self.assertEqual(
                    args[args.index("--dataset-id") + 1], sha256_file(fixture.manifest)
                )
                self.assertEqual(args[args.index("--degree") + 1], "32")
                self.assertEqual(args[args.index("--bits-block-size") + 1], "512")
                if fixture.sampling:
                    self.assertEqual(args[args.index("--seed") + 1], str(2**64 - 1))
                else:
                    self.assertNotIn("--seed", args)
                    self.assertNotIn("--temperature", args)

    def test_invalid_manifest_and_operator_flags_prevent_launch(self):
        for operator in ["token-sampling", "gradient-compression"]:
            with self.subTest(operator=operator):
                root = self.root / operator
                root.mkdir()
                fixture = TensorFixture(root, operator)
                marker = root / "launched"
                binary = root / "must-not-run"
                binary.write_text(
                    f"#!{sys.executable}\nopen({str(marker)!r}, 'w').close()\n"
                )
                binary.chmod(0o755)
                if fixture.sampling:
                    cases = [
                        ["--seed", "-1"],
                        ["--seed", str(2**64)],
                        ["--temperature", "nan"],
                        ["--temperature", "0"],
                    ]
                else:
                    cases = [["--temperature", "1"], ["--seed", "42"]]
                for options in cases:
                    run = subprocess.run(
                        fixture.launcher_command(binary) + options, capture_output=True
                    )
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse(marker.exists())
                fixture.raw.write_bytes(fixture.raw.read_bytes()[:-1])
                run = subprocess.run(
                    fixture.launcher_command(binary), capture_output=True
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(b"Wrong byte count", run.stderr)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
