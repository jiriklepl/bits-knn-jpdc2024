#!/usr/bin/env python3
"""Prepare small, middle, and large inputs for each of three applications on CPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from application_inputs import load_application_manifest, sha256_file
from model_application_inputs import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    DEFAULT_REVISION,
    GRADIENT_LOSS,
)

REPO = Path(__file__).resolve().parents[1]
PROMPT_ORIGIN = (
    "application scaling prompt pool v1; "
    "fixed sentences plus distinct template combinations"
)


def scaling_prompts(batch_size):
    """Nested prompt sets: baseline eight plus distinct prose templates."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 512
    ):
        raise ValueError("batch_size must be an integer between 1 and 512")
    people = [
        "the researcher",
        "the teacher",
        "the engineer",
        "the student",
        "the writer",
        "the traveler",
        "the artist",
        "the gardener",
    ]
    actions = [
        "examined a detailed map",
        "wrote a short explanation",
        "noticed a small mistake",
        "described an unusual discovery",
        "prepared a list of questions",
        "remembered an earlier conversation",
        "organized the available materials",
        "discussed the next experiment",
    ]
    settings = [
        "After breakfast",
        "Before the meeting",
        "During the afternoon",
        "On a rainy morning",
        "At the end of the day",
        "Following a long journey",
        "While waiting for a friend",
        "On the first day of spring",
    ]
    prompts = list(DEFAULT_PROMPTS)
    # Vary the subject fastest so every small batch samples several subjects.
    for setting in settings:
        for action in actions:
            for person in people:
                prompts.append(f"{setting}, {person} {action}.")
    return prompts[:batch_size]


def workload_specs(baseline_root, output):
    specs = []
    for tier, suffix, factor in [
        ("small", "01", 0.1),
        ("middle", "1", 1),
        ("large", "10", 10),
    ]:
        identity = f"database-sf{suffix}"
        specs.append(
            {
                "id": identity,
                "operator": "database-topn",
                "size_tier": tier,
                "label": f"TPC-H SF {factor:g}",
                "path": baseline_root / "tpch-sf01"
                if factor == 0.1
                else output / identity,
                "scale_factor": factor,
            }
        )
    for tier, batch in [("small", 8), ("middle", 128), ("large", 512)]:
        identity = f"sampling-b{batch}"
        specs.append(
            {
                "id": identity,
                "operator": "token-sampling",
                "size_tier": tier,
                "label": f"Batch {batch}",
                "path": baseline_root / "token-sampling"
                if batch == 8
                else output / identity,
                "batch_size": batch,
            }
        )
    for tier, suffix, parameter, label in [
        (
            "small",
            "attention",
            "transformer.h.0.attn.c_proj.weight",
            "Attention projection weight",
        ),
        ("middle", "mlp", "transformer.h.0.mlp.c_fc.weight", "MLP weight"),
        ("large", "embedding", "transformer.wte.weight", "Token embedding weight"),
    ]:
        identity = f"gradient-{suffix}"
        specs.append(
            {
                "id": identity,
                "operator": "gradient-compression",
                "size_tier": tier,
                "label": label,
                "path": baseline_root / "gradient-compression"
                if suffix == "mlp"
                else output / identity,
                "parameter": parameter,
            }
        )
    return specs


def index_workload(spec, suite_directory):
    path = spec["path"] / "manifest.json"
    manifest, _, dataset_id = load_application_manifest(path)
    if manifest["operator"] != spec["operator"]:
        raise ValueError(f"{path}: unexpected operator")
    source = manifest["source"]
    entry = {
        "id": spec["id"],
        "operator": spec["operator"],
        "size_tier": spec["size_tier"],
        "label": spec["label"],
        "manifest": os.path.relpath(path.resolve(), suite_directory.resolve()),
        "dataset_id": dataset_id,
    }
    if spec["operator"] == "database-topn":
        if source.get("scale_factor") != spec["scale_factor"]:
            raise ValueError(f"{path}: unexpected TPC-H scale factor")
        entry.update(
            size=manifest["rows"],
            size_unit="rows",
            rows=manifest["rows"],
            score_bytes=manifest["rows"] * 4,
            scale_factor=spec["scale_factor"],
        )
    else:
        if (
            source.get("model") != DEFAULT_MODEL
            or source.get("revision") != DEFAULT_REVISION
        ):
            raise ValueError(f"{path}: expected the pinned DistilGPT-2 model")
        if spec["operator"] == "token-sampling":
            if manifest["batch_size"] != spec["batch_size"] or source.get(
                "prompts"
            ) != scaling_prompts(spec["batch_size"]):
                raise ValueError(f"{path}: unexpected prompt pool or batch size")
            entry.update(
                size=manifest["batch_size"],
                size_unit="batch",
                batch_size=manifest["batch_size"],
                vocabulary_size=manifest["vocabulary_size"],
                score_bytes=manifest["batch_size"] * manifest["vocabulary_size"] * 4,
            )
        else:
            if source.get("parameter") != spec["parameter"]:
                raise ValueError(f"{path}: unexpected complete gradient parameter")
            if source.get("prompts") != DEFAULT_PROMPTS:
                raise ValueError(f"{path}: unexpected gradient prompt pool")
            if source.get("loss") != GRADIENT_LOSS:
                raise ValueError(f"{path}: unexpected gradient loss semantics")
            entry.update(
                size=manifest["elements"],
                size_unit="elements",
                elements=manifest["elements"],
                tensor_shape=manifest["tensor_shape"],
                parameter=spec["parameter"],
                score_bytes=manifest["elements"] * 4,
            )
    return entry


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    exporter = Path(__file__).with_name("export-application-inputs.py")
    entries = []
    for spec in workload_specs(args.baseline_root.resolve(), output):
        manifest_path = spec["path"] / "manifest.json"
        if not manifest_path.exists():
            database = spec["operator"] == "database-topn"
            command = [
                str(args.database_python if database else args.model_python),
                str(exporter),
                spec["operator"],
                "--output",
                str(spec["path"]),
            ]
            if database:
                command += ["--scale-factor", str(spec["scale_factor"])]
                if args.extension_directory:
                    command += ["--extension-directory", str(args.extension_directory)]
            else:
                command += ["--threads", str(args.threads)]
                if args.cache_directory:
                    command += ["--cache-directory", str(args.cache_directory)]
                if args.local_files_only:
                    command += ["--local-files-only"]
                if spec["operator"] == "token-sampling":
                    prompts = output / f"{spec['id']}-prompts.json"
                    prompts.write_text(
                        json.dumps(scaling_prompts(spec["batch_size"]), indent=2) + "\n"
                    )
                    command += [
                        "--prompts-file",
                        str(prompts),
                        "--prompt-origin",
                        PROMPT_ORIGIN,
                        "--microbatch-size",
                        str(args.microbatch_size),
                    ]
                else:
                    command += ["--parameter", spec["parameter"]]
            print(f"Preparing {spec['id']}", flush=True)
            subprocess.run(command, check=True)
        entry = index_workload(spec, output)
        entries.append(entry)
        print(
            f"Validated {entry['id']}: {entry['size']} {entry['size_unit']}", flush=True
        )
    suite = {
        "version": 1,
        "study": "application-size-and-parallelism",
        "prompt_pool": PROMPT_ORIGIN,
        "preparation_sha256": sha256_file(__file__),
        "workloads": entries,
    }
    temporary = output / ".suite.json.tmp"
    temporary.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "suite.json")
    return output / "suite.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPO / "data/application-inputs/scaling"
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=REPO / "data/application-inputs"
    )
    parser.add_argument("--database-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--extension-directory", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.microbatch_size <= 0:
        parser.error("threads and microbatch-size must be positive")
    try:
        print(prepare(args))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
