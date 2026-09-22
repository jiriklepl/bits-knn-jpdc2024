"""Validate saved fixed-k application studies before plotting their cases."""

import importlib.util
import hashlib
import json
from pathlib import Path

from application_inputs import sha256_file
import tensor_analysis


OPERATORS = ("database-topn", "token-sampling", "gradient-compression")
CONFIG = ("degree", "block_size", "items_per_thread")


def expected_configurations(settings, ks):
    """Match the study matrix and the native comparison backends' reported settings."""
    expected = set()
    for k in ks:
        expected.update(
            (backend, k, degree, block, items)
            for backend, degrees in (
                ("bits-prefetch", (1,)),
                ("bits-sq", settings["degrees"]),
            )
            for degree in degrees
            for block in settings["blocks"]
            for items in settings["items"]
        )
        expected.update((("air-topk", k, 1, 512, 0), ("grid-select", k, 1, 0, 0)))
        if k in (32, 64, 128, 256, 512, 1024):
            queue = 2 if k == 32 else 3 if k <= 128 else 4 if k == 256 else 8
            expected.add(("block-select", k, 1, 128, queue))
    return expected


def validate_run(summary, run, workload, settings, path):
    """Reject shape, scenario, or matrix mismatches even when raw hashes agree."""
    operator = workload["operator"]
    rows = {
        "database-topn": workload.get("rows"),
        "token-sampling": workload.get("vocabulary_size"),
        "gradient-compression": workload.get("elements"),
    }[operator]
    batch = workload["batch_size"] if operator == "token-sampling" else 1
    unit = {
        "database-topn": "rows",
        "token-sampling": "batch",
        "gradient-compression": "elements",
    }[operator]
    if (workload["size"], workload["size_unit"], workload["score_bytes"]) != (
        batch if operator == "token-sampling" else rows,
        unit,
        4 * rows * batch,
    ):
        raise ValueError(f"Workload dimensions are inconsistent: {workload['id']}")
    ks = settings["ks"]
    if set(run["ks"]) != set(ks) or len(run["ks"]) != len(ks):
        raise ValueError(f"Planned k values disagree with study settings: {path}")
    if {row["dataset_id"] for row in summary} != {workload["dataset_id"]}:
        raise ValueError(f"Timing dataset does not match workload: {path}")
    if {row["k"] for row in summary} != set(ks):
        raise ValueError(f"Timing k values do not match planned run: {path}")
    for row in summary:
        if (row["rows"], row.get("batch_size", 1)) != (rows, batch):
            raise ValueError(f"Timing shape does not match workload: {path}")
        if operator == "token-sampling" and (row["temperature"], row["seed"]) != (
            1,
            42,
        ):
            raise ValueError(f"Sampling settings differ from the scaling study: {path}")
        if row["samples"] != (
            1 if row["phase"] == "upload_shared" else settings["repeat"]
        ):
            raise ValueError(f"Timing repetition count differs from study: {path}")
        if row["speedup_vs_air"] == "":
            raise ValueError(f"Missing unique AIR baseline: {path}")
    observed = {
        tuple(row[name] for name in ("backend", "k") + CONFIG)
        for row in summary
        if row["phase"] == "operator"
    }
    if observed != expected_configurations(settings, ks):
        raise ValueError(
            f"Timing configurations do not cover the planned sweep: {path}"
        )
    if run["configurations"] != len(observed):
        raise ValueError(
            f"Configuration count differs from completed study run: {path}"
        )


def load_study(index_path):
    """Check provenance and completeness before combining any timing files."""
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text())
    if index.get("version") != 1:
        raise ValueError(f"{index_path}: unsupported study version")
    settings = index["settings"]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if digest != index["settings_sha256"]:
        raise ValueError("Scaling settings checksum mismatch")
    for field in ("ks", "degrees", "blocks", "items"):
        values = settings[field]
        if (
            not values
            or len(values) != len(set(values))
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in values
            )
        ):
            raise ValueError(f"Invalid scaling settings: {field}")
    if (
        not isinstance(settings["repeat"], int)
        or isinstance(settings["repeat"], bool)
        or settings["repeat"] < 1
    ):
        raise ValueError("Invalid scaling repetition count")
    if settings.get("retention_ratios"):
        raise ValueError("Application studies support fixed-k runs only")
    workloads = {item["id"]: item for item in index["workloads"]}
    if len(workloads) != len(index["workloads"]):
        raise ValueError("Duplicate workload ID in scaling index")
    if not index.get("runs"):
        raise ValueError("Scaling study has no runs")
    spec = importlib.util.spec_from_file_location(
        "scaling_database_analysis", Path(__file__).with_name("plot-database-topn.py")
    )
    database = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database)
    result, seen, sources = [], set(), set()
    for run in index["runs"]:
        if run["status"] != "complete":
            raise ValueError(
                f"Incomplete study: {run['workload_id']} / {run['mode']} "
                f"is {run['status']}: {run.get('reason', '')}"
            )
        workload = workloads[run["workload_id"]]
        operator = workload["operator"]
        if run["mode"] != "fixed-k" or run.get("requested_retention") is not None:
            raise ValueError("Application studies support fixed-k runs only")
        key = workload["id"]
        if key in seen:
            raise ValueError(f"Duplicate study run: {key}")
        seen.add(key)
        path = (index_path.parent / run["csv"]).resolve()
        if path in sources:
            raise ValueError(f"Multiple study runs refer to one timing CSV: {path}")
        sources.add(path)
        if sha256_file(path) != run["csv_sha256"]:
            raise ValueError(f"Timing CSV checksum mismatch: {path}")
        summary = (
            database.summarize(path)
            if operator == "database-topn"
            else tensor_analysis.summarize(path, operator)
        )
        validate_run(summary, run, workload, settings, path)
        for row in summary:
            result.append(
                row
                | {
                    "operator": operator,
                    "workload_id": workload["id"],
                    "size_tier": workload.get("size_tier", ""),
                    "mode": run["mode"],
                    "source_csv": str(path),
                }
            )
    # A missing run entry must not silently turn a partial study into a full one.
    if seen != set(workloads):
        raise ValueError("Study index does not cover every workload")
    return result
