"""Run with: python3 -m unittest discover -s tests/applications -p 'test_*.py'."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_inputs import (  # noqa: E402
    COLUMN_TYPES,
    SEMANTICS,
    load_database_manifest,
    sha256_file,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            "version": 1,
            "operator": "database-topn",
            "rows": 2,
            "semantics": SEMANTICS,
            "source": {"kind": "synthetic-test"},
            "columns": {},
        }
        for name, (dtype, width) in COLUMN_TYPES.items():
            path = self.root / (name + ".raw")
            path.write_bytes(
                struct.pack("<2f", 1, 2)
                if width == 4
                else struct.pack("<2Q", 2**40, 2**40 + 7)
            )
            self.manifest["columns"][name] = {
                "file": path.name,
                "dtype": dtype,
                "shape": [2],
                "byte_order": "little",
                "sha256": sha256_file(path),
            }
        self.path = self.root / "manifest.json"
        self.save(self.manifest)

    def save(self, manifest):
        self.path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_valid_manifest(self):
        manifest, columns, digest = load_database_manifest(self.path)
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(set(columns), set(COLUMN_TYPES))
        self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_rejects_schema_errors(self):
        changes = [
            ("version", 2),
            ("version", True),
            ("operator", "sampling"),
            ("rows", 0),
            ("rows", True),
            ("rows", 2.0),
            ("rows", 2**31),
            ("source", {}),
            ("semantics", {}),
            ("columns", {}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                broken = copy.deepcopy(self.manifest)
                broken[key] = value
                self.save(broken)
                with self.assertRaises(ValueError):
                    load_database_manifest(self.path)
        for value in [[], None, 1]:
            self.save(value)
            with self.assertRaises(ValueError):
                load_database_manifest(self.path)

    def test_rejects_column_metadata_errors(self):
        for key, value in [
            ("dtype", "float64"),
            ("byte_order", "big"),
            ("shape", [1, 2]),
            ("shape", [2.0]),
            ("shape", [3]),
            ("file", "/tmp/abs.raw"),
            ("sha256", "z" * 64),
            ("sha256", "0" * 64),
        ]:
            with self.subTest(key=key, value=value):
                broken = copy.deepcopy(self.manifest)
                broken["columns"]["price"][key] = value
                self.save(broken)
                with self.assertRaises(ValueError):
                    load_database_manifest(self.path)

    def test_rejects_corruption_truncation_and_extra_bytes(self):
        path = self.root / "price.raw"
        original = path.read_bytes()
        for broken in [original[:-1], original + b"\0", b"\xff" + original[1:]]:
            with self.subTest(size=len(broken)):
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    load_database_manifest(self.path)

    def test_rejects_escape_and_symlink_escape(self):
        nested = self.root / "nested"
        nested.mkdir()
        nested_path = nested / "manifest.json"
        broken = copy.deepcopy(self.manifest)
        broken["columns"]["price"]["file"] = "../price.raw"
        nested_path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            load_database_manifest(nested_path)
        (nested / "price.raw").symlink_to(self.root / "price.raw")
        broken["columns"]["price"]["file"] = "price.raw"
        nested_path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            load_database_manifest(nested_path)

    def test_runner_stops_before_launch_on_invalid_manifest(self):
        fake_binary = self.root / "should-not-run"
        marker = self.root / "launched"
        fake_binary.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        fake_binary.chmod(0o755)
        (self.root / "price.raw").write_bytes(b"bad")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "run-applications.py"),
                str(self.path),
                "--binary",
                str(fake_binary),
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertIn(b"Wrong byte count", result.stderr)

    def test_runner_propagates_failure_and_preserves_arguments(self):
        fake_binary = self.root / "fake binary"
        arguments = self.root / "arguments.json"
        fake_binary.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"open({str(arguments)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "sys.exit(7)\n"
        )
        fake_binary.chmod(0o755)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "run-applications.py"),
                str(self.path),
                "--binary",
                str(fake_binary),
                "--k",
                "1",
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 7)
        args = json.loads(arguments.read_text())
        self.assertEqual(args[args.index("--prices") + 1], str(self.root / "price.raw"))
        self.assertEqual(args[args.index("-k") + 1], "1")
        self.assertEqual(
            args[args.index("--backends") + 1],
            "bits-sq,air-topk,grid-select,block-select",
        )
        self.assertEqual(args[args.index("--degree") + 1], "32")
        self.assertEqual(args[args.index("--bits-block-size") + 1], "512")
        self.assertNotIn("--items-per-thread", args)
        self.assertEqual(args[args.index("--dataset-id") + 1], sha256_file(self.path))


if __name__ == "__main__":
    unittest.main()
