"""Set DATABASE_TOPN_BINARY to enable native GPU integration tests."""

import csv
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_inputs import COLUMN_TYPES, SEMANTICS, sha256_file  # noqa: E402


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


@unittest.skipUnless(
    os.environ.get("DATABASE_TOPN_BINARY"),
    "Set DATABASE_TOPN_BINARY for GPU integration tests",
)
class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ["DATABASE_TOPN_BINARY"]).resolve()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.n = 131
        self.columns = {
            "price": [f32((i % 17 - 8) * 1.001) for i in range(self.n)],
            "discount": [f32((i % 5) / 100) for i in range(self.n)],
            "payload": [f32(i + 0.25) for i in range(self.n)],
            "row_id": [2**40 + i * 7 for i in range(self.n)],
        }
        manifest = {
            "version": 1,
            "operator": "database-topn",
            "rows": self.n,
            "semantics": SEMANTICS,
            "source": {"kind": "synthetic-test"},
            "columns": {},
        }
        for name, (dtype, _) in COLUMN_TYPES.items():
            path = self.root / (name + ".raw")
            path.write_bytes(
                struct.pack(
                    "<" + str(self.n) + ("Q" if dtype == "uint64" else "f"),
                    *self.columns[name],
                )
            )
            manifest["columns"][name] = {
                "file": path.name,
                "dtype": dtype,
                "shape": [self.n],
                "byte_order": "little",
                "sha256": sha256_file(path),
            }
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        self.raw_command = [
            str(self.binary),
            "--rows",
            str(self.n),
            "--prices",
            str(self.root / "price.raw"),
            "--discounts",
            str(self.root / "discount.raw"),
            "--payload",
            str(self.root / "payload.raw"),
            "--row-ids",
            str(self.root / "row_id.raw"),
        ]

    def test_complete_comparison_and_timing_schema(self):
        output = self.root / "rows.csv"
        command = [
            sys.executable,
            str(SCRIPTS / "run-applications.py"),
            str(self.manifest),
            "--binary",
            str(self.binary),
            "--k",
            "32",
            "--backends",
            "bits,bits-sq,air-topk,grid-select,block-select",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output",
            str(output),
        ]
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        timings = list(csv.DictReader(io.StringIO(run.stdout)))
        self.assertEqual(len(timings), 5 * (1 + 2 * 5))
        backends = {"bits", "bits-sq", "air-topk", "grid-select", "block-select"}
        self.assertEqual({r["backend"] for r in timings}, backends)
        self.assertEqual(
            {r["phase"] for r in timings},
            {
                "operator",
                "download",
                "upload_shared",
                "transform_isolated",
                "selection_isolated",
                "output_isolated",
            },
        )
        for row in timings:
            self.assertEqual(row["dataset_id"], sha256_file(self.manifest))
            self.assertGreaterEqual(float(row["seconds"]), 0)
            self.assertAlmostEqual(float(row["retention_ratio"]), 32 / self.n)
            expected_config = {
                "bits": (1, 512, 7),
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
                expected_config,
            )
        scores = [
            f32(p * f32(1 - d))
            for p, d in zip(self.columns["price"], self.columns["discount"])
        ]
        expected = sorted(scores, reverse=True)[:32]
        with output.open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 5 * 32)
        for backend in backends:
            selected = [r for r in rows if r["backend"] == backend]
            self.assertEqual([f32(float(r["score"])) for r in selected], expected)
            self.assertEqual(len({r["source_index"] for r in selected}), 32)
            for row in selected:
                index = int(row["source_index"])
                self.assertTrue(0 <= index < self.n)
                self.assertEqual(int(row["row_id"]), self.columns["row_id"][index])
                self.assertEqual(f32(float(row["score"])), scores[index])
                self.assertEqual(float(row["payload"]), self.columns["payload"][index])

    def test_explicit_bits_configuration_is_preserved(self):
        for block, batch in [(128, 1), (256, 4), (512, 16)]:
            with self.subTest(block=block, batch=batch):
                run = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "run-applications.py"),
                        str(self.manifest),
                        "--binary",
                        str(self.binary),
                        "--backends",
                        "bits,bits-sq",
                        "--bits-block-size",
                        str(block),
                        "--items-per-thread",
                        str(batch),
                        "--degree",
                        "7",
                        "--warmup",
                        "1",
                        "--repeat",
                        "1",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                for row in csv.DictReader(io.StringIO(run.stdout)):
                    self.assertEqual(int(row["block_size"]), block)
                    self.assertEqual(int(row["items_per_thread"]), batch)
                    self.assertEqual(
                        int(row["degree"]), 7 if row["backend"] == "bits-sq" else 1
                    )

    def test_invalid_configurations_fail(self):
        for options in [
            ["-k", "0"],
            ["-k", "132"],
            ["-k", "7"],
            ["--degree", "0"],
            ["--degree", "132"],
            ["--items-per-thread", "2"],
            ["--items-per-thread", "0"],
            ["--bits-block-size", "0"],
            ["--bits-block-size", "129"],
            ["--bits-block-size", "1024"],
            ["--backends", "missing"],
            ["--backends", "bits,bits"],
            ["--backends", "bits,"],
            ["--repeat", "0"],
            ["-k", "32garbage"],
        ]:
            with self.subTest(options=options):
                run = subprocess.run(
                    self.raw_command + options, text=True, capture_output=True
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr.strip())

    def test_nonfinite_raw_values_and_wrong_file_size_fail(self):
        path = self.root / "price.raw"
        original = path.read_bytes()
        for content in [
            original[:-1],
            original + b"\0",
            struct.pack("<f", float("nan")) + original[4:],
        ]:
            path.write_bytes(content)
            run = subprocess.run(self.raw_command, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
        path.write_bytes(original)

    def test_small_arbitrary_k_without_split_backend(self):
        # The unused default split degree must not reject a one-row query.
        for name, (dtype, _) in COLUMN_TYPES.items():
            (self.root / (name + ".raw")).write_bytes(
                struct.pack("<Q" if dtype == "uint64" else "<f", self.columns[name][0])
            )
        command = self.raw_command.copy()
        command[command.index("--rows") + 1] = "1"
        run = subprocess.run(
            command
            + [
                "-k",
                "1",
                "--backends",
                "air-topk,grid-select",
                "--warmup",
                "0",
                "--repeat",
                "1",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_default_comparison_includes_grid_select(self):
        run = subprocess.run(
            self.raw_command + ["--warmup", "0", "--repeat", "1"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(
            {row["backend"] for row in csv.DictReader(io.StringIO(run.stdout))},
            {"bits-sq", "air-topk", "grid-select", "block-select"},
        )


if __name__ == "__main__":
    unittest.main()
