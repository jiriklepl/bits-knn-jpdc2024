"""Scaling-suite provenance and distinct nested prompts, without model downloads."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from application_inputs import (  # noqa: E402
    GRADIENT_SEMANTICS,
    SAMPLING_SEMANTICS,
    sha256_file,
)
from model_application_inputs import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    DEFAULT_PROMPTS,
    GRADIENT_LOSS,
)

spec = importlib.util.spec_from_file_location(
    "prepare_scaling", SCRIPTS / "prepare-application-scaling.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ScalingInputTests(unittest.TestCase):
    def test_gradient_reuse_rejects_changed_prompts_or_loss_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = root / "gradient.f32"
            values.write_bytes(struct.pack("<6f", -1, 2, 3, -4, 5, 6))
            parameter = "transformer.wte.weight"
            manifest = {
                "version": 1,
                "operator": "gradient-compression",
                "elements": 6,
                "tensor_shape": [2, 3],
                "semantics": GRADIENT_SEMANTICS,
                "source": {
                    "model": DEFAULT_MODEL,
                    "revision": DEFAULT_REVISION,
                    "parameter": parameter,
                    "prompts": list(DEFAULT_PROMPTS),
                    "loss": GRADIENT_LOSS,
                },
                "columns": {
                    "gradient": {
                        "file": values.name,
                        "dtype": "float32",
                        "byte_order": "little",
                        "shape": [2, 3],
                        "sha256": sha256_file(values),
                    }
                },
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            workload = {
                "id": "gradient-embedding",
                "operator": "gradient-compression",
                "label": "Token embedding weight",
                "size_tier": "large",
                "path": root,
                "parameter": parameter,
            }
            entry = prepare.index_workload(workload, root)
            self.assertEqual(entry["dataset_id"], sha256_file(path))
            self.assertEqual(entry["size_tier"], "large")
            original_hash = sha256_file(values)
            changes = [
                (
                    "prompts",
                    ["A different prompt.", *DEFAULT_PROMPTS[1:]],
                    "prompt pool",
                ),
                ("prompts", None, "prompt pool"),
                ("loss", "sum causal next-token cross entropy", "loss semantics"),
                ("loss", None, "loss semantics"),
            ]
            for key, value, message in changes:
                with self.subTest(key=key, value=value):
                    previous = manifest["source"][key]
                    manifest["source"][key] = value
                    path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, message):
                        prepare.index_workload(workload, root)
                    manifest["source"][key] = previous
                    self.assertEqual(sha256_file(values), original_hash)

    def test_prompt_pool_is_distinct_nested_and_preserves_baseline(self):
        prompts = prepare.scaling_prompts(512)
        self.assertEqual(len(prompts), 512)
        self.assertEqual(len(set(prompts)), 512)
        self.assertEqual(prompts[:8], DEFAULT_PROMPTS)
        self.assertTrue(all(prompt.strip() for prompt in prompts))
        for batch in [8, 32, 128, 512]:
            self.assertEqual(prepare.scaling_prompts(batch), prompts[:batch])
        self.assertEqual(prepare.scaling_prompts(512), prompts)
        for batch in [0, 513, True, 8.0, "8"]:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                prepare.scaling_prompts(batch)

    def test_nine_workloads_cover_requested_scales_and_complete_parameters(self):
        specs = prepare.workload_specs(Path("baseline"), Path("scaling"))
        self.assertEqual(len(specs), 9)
        self.assertEqual(len({spec["id"] for spec in specs}), 9)
        for operator in ["database-topn", "token-sampling", "gradient-compression"]:
            with self.subTest(operator=operator):
                self.assertEqual(
                    [s["size_tier"] for s in specs if s["operator"] == operator],
                    ["small", "middle", "large"],
                )
        self.assertEqual(
            [s["scale_factor"] for s in specs if s["operator"] == "database-topn"],
            [0.1, 1, 10],
        )
        self.assertEqual(
            [s["batch_size"] for s in specs if s["operator"] == "token-sampling"],
            [8, 128, 512],
        )
        self.assertEqual(
            [s["parameter"] for s in specs if s["operator"] == "gradient-compression"],
            [
                "transformer.h.0.attn.c_proj.weight",
                "transformer.h.0.mlp.c_fc.weight",
                "transformer.wte.weight",
            ],
        )
        self.assertEqual(
            {s["id"]: s["path"] for s in specs},
            {
                "database-sf01": Path("baseline/tpch-sf01"),
                "database-sf1": Path("scaling/database-sf1"),
                "database-sf10": Path("scaling/database-sf10"),
                "sampling-b8": Path("baseline/token-sampling"),
                "sampling-b128": Path("scaling/sampling-b128"),
                "sampling-b512": Path("scaling/sampling-b512"),
                "gradient-attention": Path("scaling/gradient-attention"),
                "gradient-mlp": Path("baseline/gradient-compression"),
                "gradient-embedding": Path("scaling/gradient-embedding"),
            },
        )

    def test_index_checks_data_hash_prompt_pool_and_pinned_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "sampling-b128"
            data.mkdir()
            values = data / "logits.f32"
            values.write_bytes(struct.pack("<384f", *range(384)))
            manifest = {
                "version": 1,
                "operator": "token-sampling",
                "batch_size": 128,
                "vocabulary_size": 3,
                "semantics": SAMPLING_SEMANTICS,
                "source": {
                    "model": DEFAULT_MODEL,
                    "revision": DEFAULT_REVISION,
                    "prompts": prepare.scaling_prompts(128),
                },
                "columns": {
                    "logits": {
                        "file": "logits.f32",
                        "dtype": "float32",
                        "byte_order": "little",
                        "shape": [128, 3],
                        "sha256": sha256_file(values),
                    }
                },
            }
            path = data / "manifest.json"
            path.write_text(json.dumps(manifest))
            workload = {
                "id": "sampling-b128",
                "operator": "token-sampling",
                "label": "Batch 128",
                "size_tier": "middle",
                "path": data,
                "batch_size": 128,
            }
            entry = prepare.index_workload(workload, root)
            self.assertEqual(entry["manifest"], "sampling-b128/manifest.json")
            self.assertEqual(entry["dataset_id"], sha256_file(path))
            self.assertEqual(entry["size_tier"], "middle")
            self.assertEqual(
                (entry["size"], entry["size_unit"], entry["score_bytes"]),
                (128, "batch", 1536),
            )
            manifest["source"]["prompts"][0] = "A different prompt."
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unexpected prompt pool"):
                prepare.index_workload(workload, root)
            manifest["source"]["prompts"] = prepare.scaling_prompts(128)
            manifest["source"]["revision"] = "a" * 40
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "pinned DistilGPT-2"):
                prepare.index_workload(workload, root)
            manifest["source"]["revision"] = DEFAULT_REVISION
            path.write_text(json.dumps(manifest))
            values.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "Wrong byte count"):
                prepare.index_workload(workload, root)


if __name__ == "__main__":
    unittest.main()
