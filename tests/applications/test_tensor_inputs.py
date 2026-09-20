"""Tensor manifests and CLI validation; no model downloads or CUDA required."""

import copy
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
    GRADIENT_SEMANTICS,
    SAMPLING_SEMANTICS,
    load_application_manifest,
    load_gradient_manifest,
    load_sampling_manifest,
    sha256_file,
)
from model_application_inputs import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    DEFAULT_REVISION,
    export_model_input,
    validate_capture_options,
)


class TensorManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "manifest.json"

    def fixture(self, operator="token-sampling"):
        sampling = operator == "token-sampling"
        name = "logits" if sampling else "gradient"
        array = self.root / f"{name}.f32"
        array.write_bytes(struct.pack("<6f", -1, 0, 2, 3, -4, 5))
        manifest = {
            "version": 1,
            "operator": operator,
            "semantics": SAMPLING_SEMANTICS if sampling else GRADIENT_SEMANTICS,
            "source": {"kind": "synthetic-unit-test"},
            "columns": {
                name: {
                    "file": array.name,
                    "dtype": "float32",
                    "byte_order": "little",
                    "shape": [2, 3],
                    "sha256": sha256_file(array),
                }
            },
        }
        if sampling:
            manifest.update(batch_size=2, vocabulary_size=3)
        else:
            manifest.update(elements=6, tensor_shape=[2, 3])
        self.save(manifest)
        return manifest

    def save(self, manifest):
        self.path.write_text(json.dumps(manifest), encoding="utf-8")

    def runner(self, *args):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "run-applications.py"),
                str(self.path),
                "--binary",
                str(self.fake_binary),
                *args,
            ],
            capture_output=True,
            text=True,
        )

    def fake_program(self):
        self.fake_binary = self.root / "fake application"
        self.arguments = self.root / "arguments.json"
        self.fake_binary.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"open({str(self.arguments)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "sys.exit(7)\n"
        )
        self.fake_binary.chmod(0o755)

    def test_valid_tensor_manifests(self):
        for operator, loader in [
            ("token-sampling", load_sampling_manifest),
            ("gradient-compression", load_gradient_manifest),
        ]:
            with self.subTest(operator=operator):
                expected = self.fixture(operator)
                manifest, columns, digest = loader(self.path)
                self.assertEqual(manifest, expected)
                self.assertEqual(digest, sha256_file(self.path))
                self.assertEqual(set(columns), set(expected["columns"]))
                self.assertEqual(
                    load_application_manifest(self.path), loader(self.path)
                )

    def test_rejects_operator_schema_mismatch(self):
        self.fixture()
        with self.assertRaises(ValueError):
            load_gradient_manifest(self.path)
        for value in [[], None, 1, {"operator": []}, {"operator": "unknown"}]:
            self.save(value)
            with self.assertRaises(ValueError):
                load_application_manifest(self.path)

    def test_rejects_invalid_dimensions_and_semantics(self):
        common = [
            ("version", True),
            ("version", 2),
            ("source", {}),
            ("semantics", {}),
            ("columns", {}),
        ]
        for operator in ["token-sampling", "gradient-compression"]:
            original = self.fixture(operator)
            fields = (
                ["batch_size", "vocabulary_size"]
                if operator == "token-sampling"
                else ["elements"]
            )
            changes = common + [
                (field, value)
                for field in fields
                for value in [True, 0, -1, 2.0, 2**31]
            ]
            if operator == "gradient-compression":
                changes += [
                    ("tensor_shape", value)
                    for value in [None, [], [6.0], [True, 6], [2, 4], [0, 6], [-1, -6]]
                ]
            for key, value in changes:
                with self.subTest(operator=operator, key=key, value=value):
                    broken = copy.deepcopy(original)
                    broken[key] = value
                    self.save(broken)
                    with self.assertRaises(ValueError):
                        load_application_manifest(self.path)

    def test_rejects_column_metadata_and_corruption(self):
        for operator, name in [
            ("token-sampling", "logits"),
            ("gradient-compression", "gradient"),
        ]:
            original = self.fixture(operator)
            for key, value in [
                ("dtype", "float64"),
                ("byte_order", "big"),
                ("shape", [6]),
                ("shape", [2.0, 3]),
                ("file", "/tmp/escape.f32"),
                ("sha256", "z" * 64),
                ("sha256", "0" * 64),
            ]:
                with self.subTest(operator=operator, key=key):
                    broken = copy.deepcopy(original)
                    broken["columns"][name][key] = value
                    self.save(broken)
                    with self.assertRaises(ValueError):
                        load_application_manifest(self.path)
            self.save(original)
            array = self.root / f"{name}.f32"
            data = array.read_bytes()
            for corrupted in [data[:-1], data + b"\x00", b"\xff" + data[1:]]:
                array.write_bytes(corrupted)
                with self.assertRaises(ValueError):
                    load_application_manifest(self.path)

    def test_rejects_column_symlink_escape(self):
        manifest = self.fixture()
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "logits.f32").symlink_to(self.root / "logits.f32")
        path = nested / "manifest.json"
        for filename in ["logits.f32", "../logits.f32"]:
            manifest["columns"]["logits"]["file"] = filename
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_sampling_manifest(path)

    def test_sampling_runner_arguments_and_failure_propagation(self):
        self.fixture()
        self.fake_program()
        result = self.runner(
            "--temperature", "0.7", "--seed", str(2**64 - 1), "--k", "2"
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        args = json.loads(self.arguments.read_text())
        for flag, expected in [
            ("--logits", str(self.root / "logits.f32")),
            ("--batch-size", "2"),
            ("--vocabulary-size", "3"),
            ("--temperature", "0.7"),
            ("--seed", str(2**64 - 1)),
            ("-k", "2"),
            ("--degree", "32"),
            ("--dataset-id", sha256_file(self.path)),
        ]:
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertNotIn("--rows", args)

    def test_sampling_runner_defaults(self):
        self.fixture()
        self.fake_program()
        self.assertEqual(self.runner().returncode, 7)
        args = json.loads(self.arguments.read_text())
        self.assertEqual(args[args.index("--temperature") + 1], "1.0")
        self.assertEqual(args[args.index("--seed") + 1], "42")

    def test_sampling_runner_rejects_bad_configuration_before_launch(self):
        self.fixture()
        self.fake_program()
        for args in [
            ("--temperature", "0"),
            ("--temperature", "nan"),
            ("--temperature", "inf"),
            ("--seed", "-1"),
            ("--seed", str(2**64)),
        ]:
            with self.subTest(args=args):
                self.assertNotEqual(self.runner(*args).returncode, 0)
                self.assertFalse(self.arguments.exists())

    def test_gradient_runner_rejects_sampling_options(self):
        self.fixture("gradient-compression")
        self.fake_program()
        for args in [("--temperature", "1"), ("--seed", "42")]:
            result = self.runner(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only apply to token-sampling", result.stderr)
            self.assertFalse(self.arguments.exists())
        self.assertEqual(self.runner().returncode, 7)
        args = json.loads(self.arguments.read_text())
        self.assertEqual(
            args[args.index("--gradient") + 1], str(self.root / "gradient.f32")
        )
        self.assertEqual(args[args.index("--elements") + 1], "6")
        self.assertNotIn("--temperature", args)

    def test_runner_rejects_existing_output(self):
        self.fixture()
        self.fake_program()
        result = self.runner("--output", str(self.path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertFalse(self.arguments.exists())


class ModelExportValidationTests(unittest.TestCase):
    def test_defaults_are_fixed_and_custom_models_require_revision(self):
        self.assertEqual(
            validate_capture_options(DEFAULT_MODEL, None, DEFAULT_PROMPTS, 42, 1),
            DEFAULT_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "custom model requires"):
            validate_capture_options("other/model", None, DEFAULT_PROMPTS, 42, 1)

    def test_rejects_ambiguous_revision_or_invalid_capture_before_model_import(self):
        options = [DEFAULT_MODEL, DEFAULT_REVISION, DEFAULT_PROMPTS, 42, 1]
        for index, value in [
            (1, "main"),
            (1, "v1"),
            (1, "a" * 39),
            (2, []),
            (2, [""]),
            (2, "prompt"),
            (3, True),
            (3, -1),
            (3, 2**64),
            (4, 0),
            (4, True),
        ]:
            with self.subTest(index=index, value=value):
                broken = options.copy()
                broken[index] = value
                with self.assertRaises(ValueError):
                    validate_capture_options(*broken)

    def test_existing_export_is_preserved_without_importing_model_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "marker"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                export_model_input("token-sampling", root)
            self.assertEqual(marker.read_text(), "keep")

    def test_microbatch_and_prompt_origin_validation_precedes_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "new"
            for value in [0, -1, True, 1.5, "8"]:
                with self.subTest(microbatch_size=value):
                    with self.assertRaisesRegex(ValueError, "microbatch_size"):
                        export_model_input(
                            "token-sampling", destination, microbatch_size=value
                        )
            for value in ["", "  ", 1]:
                with self.subTest(prompt_origin=value):
                    with self.assertRaisesRegex(ValueError, "prompt_origin"):
                        export_model_input(
                            "token-sampling", destination, prompt_origin=value
                        )
            self.assertFalse(destination.exists())

    def test_exporter_rejects_options_for_wrong_operator(self):
        for args in [
            ["database-topn", "--temperature", "1"],
            ["token-sampling", "--scale-factor", "0.1"],
            ["token-sampling", "--parameter", "anything"],
            ["gradient-compression", "--extension-directory", "/tmp"],
        ]:
            with self.subTest(args=args):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "export-application-inputs.py"),
                        *args,
                        "--output",
                        "/tmp/unused-export-test",
                    ],
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"unrecognized arguments", result.stderr)

    def test_explicit_null_prompts_are_not_replaced_with_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            prompts = Path(root) / "prompts.json"
            prompts.write_text("null")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "export-application-inputs.py"),
                    "token-sampling",
                    "--prompts-file",
                    str(prompts),
                    "--output",
                    str(Path(root) / "output"),
                ],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"prompts must be a nonempty JSON list", result.stderr)


if __name__ == "__main__":
    unittest.main()
