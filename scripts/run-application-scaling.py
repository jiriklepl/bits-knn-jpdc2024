#!/usr/bin/env python3
"""Run application sizes sequentially, with resumable workload checkpoints."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

from application_inputs import load_application_manifest, sha256_file

SCRIPTS = Path(__file__).resolve().parent
OPERATORS = ("database-topn", "token-sampling", "gradient-compression")
BLOCKS = (128, 256, 512)
ITEMS = (4, 7, 8, 13, 16)
DEFAULT_KS = (32, 64, 128, 256, 512, 1024)
DEFAULT_DEGREES = (8, 32, 128, 512)
SIZE_TIERS = ("small", "middle", "large")
PHASES = (
    "operator",
    "download",
    "transform_isolated",
    "selection_isolated",
    "output_isolated",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def gpu_identity():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout), skipinitialspace=True))
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError("Cannot identify benchmark GPUs using nvidia-smi")
    return [dict(zip(("index", "uuid", "name", "driver_version"), row)) for row in rows]


def ensure_gpu_idle(gpus):
    """Reject concurrent compute before each workload using the active-process list."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    # UUID-only visibility can be mapped exactly. Numeric CUDA device ordinals may
    # differ from nvidia-smi indices, so conservatively check all inventoried GPUs.
    selected = {gpu["uuid"] for gpu in gpus}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",")]
        if all(token.startswith("GPU-") for token in tokens):
            selected = {
                uuid
                for uuid in selected
                if any(uuid.startswith(token) for token in tokens)
            }
    active = []
    for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError(
                "Cannot check active GPU compute processes using nvidia-smi"
            )
        if row[0] in selected:
            active.append(f"{row[0]}: PID {row[1]} ({row[2]})")
    if active:
        raise ValueError(
            "GPU compute is already active; refusing contended measurements: "
            + "; ".join(active)
        )


def write_json(path, value):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def unique_positive(values, name):
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {name}")


def configurations(ks, degrees):
    """Plain bits once/pair, split once/triple, comparison backends once/k."""
    for k in ks:
        for block in BLOCKS:
            for items in ITEMS:
                for degree in degrees:
                    backends = ["bits-sq"]
                    if degree == degrees[0]:
                        backends.insert(0, "bits-prefetch")
                        if (block, items) == (BLOCKS[-1], ITEMS[-1]):
                            backends += ["air-topk", "grid-select"]
                            if k in DEFAULT_KS:
                                backends.append("block-select")
                    yield dict(
                        k=k, block=block, items=items, degree=degree, backends=backends
                    )


def load_suite(path):
    suite = json.loads(path.read_text())
    if suite.get("version") != 1 or not isinstance(suite.get("workloads"), list):
        raise ValueError("Expected a scaling suite with version 1 and workloads")
    if not suite["workloads"]:
        raise ValueError("Scaling suite has no workloads")
    seen = set()
    seen_tiers = set()
    manifests = {}
    for workload in suite["workloads"]:
        identity = workload.get("id", "")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity)
            or identity in seen
        ):
            raise ValueError(f"Invalid or duplicate workload id: {identity}")
        seen.add(identity)
        relative = Path(workload["manifest"])
        if relative.is_absolute():
            raise ValueError(f"Expected a suite-relative manifest: {identity}")
        manifest_path = (path.parent / relative).resolve(strict=True)
        manifest, _, digest = load_application_manifest(manifest_path)
        operator = workload["operator"]
        if operator not in OPERATORS or manifest["operator"] != operator:
            raise ValueError(f"Operator mismatch: {identity}")
        if "size_tier" in workload:
            tier = workload["size_tier"]
            if tier not in SIZE_TIERS:
                raise ValueError(f"Invalid size tier for {identity}: {tier}")
            if (operator, tier) in seen_tiers:
                raise ValueError(f"Duplicate size tier for {operator}: {tier}")
            seen_tiers.add((operator, tier))
        if workload["dataset_id"] != digest:
            raise ValueError(f"Manifest SHA-256 mismatch: {identity}")
        size_name, size_unit = {
            "database-topn": ("rows", "rows"),
            "token-sampling": ("batch_size", "batch"),
            "gradient-compression": ("elements", "elements"),
        }[operator]
        count = manifest[size_name]
        score_bytes = 4 * count
        if operator == "token-sampling":
            score_bytes *= manifest["vocabulary_size"]
        if (workload["size"], workload["size_unit"], workload["score_bytes"]) != (
            count,
            size_unit,
            score_bytes,
        ):
            raise ValueError(f"Workload dimensions disagree with manifest: {identity}")
        if not isinstance(workload.get("label"), str) or not workload["label"]:
            raise ValueError(f"Workload label is required: {identity}")
        manifests[identity] = (manifest_path, manifest)
    return suite, manifests


def make_runs(workloads, manifests, ks, degrees, ratios):
    runs = []
    stems = set()
    for workload in workloads:
        identity = workload["id"]
        operator = workload["operator"]
        manifest = manifests[identity][1]
        candidates = manifest[
            {
                "database-topn": "rows",
                "token-sampling": "vocabulary_size",
                "gradient-compression": "elements",
            }[operator]
        ]
        modes = [("fixed-k", None, ks)]
        if operator == "gradient-compression":
            for ratio in ratios:
                k = max(
                    1,
                    int(
                        (Decimal(str(ratio)) * candidates).to_integral_value(
                            rounding=ROUND_HALF_UP
                        )
                    ),
                )
                modes.append(("retention", ratio, [k]))
        for mode, ratio, selected_ks in modes:
            suffix = "fixed-k" if ratio is None else f"retention-{ratio}"
            if "size_tier" in workload:
                stem = f"{operator}-{workload['size_tier']}"
                if ratio is not None:
                    stem += f"-{suffix}"
            else:
                stem = f"{identity}-{suffix}"
            if stem in stems:
                raise ValueError(f"Study output filenames collide: {stem}")
            stems.add(stem)
            record = dict(
                workload_id=identity,
                mode=mode,
                requested_retention=ratio,
                ks=list(selected_ks),
                csv=f"{stem}.csv",
                log=f"{stem}.err",
                status="pending",
            )
            if ratio is not None:
                record["actual_retention"] = selected_ks[0] / candidates
            unsupported = []
            for k in selected_ks:
                if not 1 <= k <= min(candidates, 2048):
                    unsupported.append(
                        f"k={k} requires 1 <= k <= min(candidates={candidates}, 2048)"
                    )
            if max(degrees) > candidates:
                unsupported.append(
                    f"split degree {max(degrees)} exceeds {candidates} candidates"
                )
            if unsupported:
                record.update(status="unsupported", reason="; ".join(unsupported))
            runs.append(record)
    return runs


def validate_chunk(output, configuration, dataset_id, repeat):
    reader = csv.DictReader(io.StringIO(output))
    fields = reader.fieldnames
    required = {
        "dataset_id",
        "backend",
        "k",
        "degree",
        "block_size",
        "items_per_thread",
        "iteration",
        "phase",
        "seconds",
    }
    if fields is None or not required.issubset(fields):
        raise ValueError("Native runner returned an invalid timing CSV header")
    rows = list(reader)
    observed = Counter()
    for row in rows:
        if row["dataset_id"] != dataset_id or int(row["k"]) != configuration["k"]:
            raise ValueError("Native timing row belongs to a different dataset or k")
        value = float(row["seconds"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Native timing must be finite and nonnegative")
        if row["backend"] in ("bits-prefetch", "bits-sq"):
            expected_degree = (
                1 if row["backend"] == "bits-prefetch" else configuration["degree"]
            )
            if (
                int(row["degree"]),
                int(row["block_size"]),
                int(row["items_per_thread"]),
            ) != (expected_degree, configuration["block"], configuration["items"]):
                raise ValueError("Native timing row has a different bits configuration")
        observed[(row["backend"], row["phase"], int(row["iteration"]))] += 1
    expected = Counter(
        (backend, phase, iteration)
        for backend in configuration["backends"]
        for phase in PHASES
        for iteration in range(repeat)
    )
    expected.update(
        (backend, "upload_shared", -1) for backend in configuration["backends"]
    )
    if observed != expected:
        raise ValueError("Incomplete, duplicate or unexpected native timing rows")
    return fields, rows


def run_workload(record, workload, manifest_path, binary, output_dir, settings):
    destination = output_dir / record["csv"]
    if destination.exists():
        raise ValueError(f"Refusing to overwrite unverified timing file: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.stem + "-", suffix=".partial", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="") as output, (
            output_dir / record["log"]
        ).open("w") as log:
            writer = None
            fields = None
            configurations_run = 0
            for configuration in configurations(record["ks"], settings["degrees"]):
                command = [
                    sys.executable,
                    str(SCRIPTS / "run-applications.py"),
                    str(manifest_path),
                    "--binary",
                    str(binary),
                    "--backends",
                    ",".join(configuration["backends"]),
                    "--degree",
                    str(configuration["degree"]),
                    "--bits-block-size",
                    str(configuration["block"]),
                    "--items-per-thread",
                    str(configuration["items"]),
                    "--k",
                    str(configuration["k"]),
                    "--warmup",
                    str(settings["warmup"]),
                    "--repeat",
                    str(settings["repeat"]),
                ]
                log.write(json.dumps({"started": now(), "command": command}) + "\n")
                log.flush()
                completed = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=log, text=True, check=True
                )
                chunk_fields, rows = validate_chunk(
                    completed.stdout,
                    configuration,
                    workload["dataset_id"],
                    settings["repeat"],
                )
                if writer is None:
                    fields = chunk_fields
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                elif chunk_fields != fields:
                    raise ValueError(
                        "Native timing CSV headers changed within a workload"
                    )
                writer.writerows(rows)
                configurations_run += len(configuration["backends"])
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
        return dict(
            csv_sha256=sha256_file(destination),
            configurations=configurations_run,
            completed_at=now(),
        )
    finally:
        temporary.unlink(missing_ok=True)


def execute(args):
    unique_positive(args.ks, "ks")
    unique_positive(args.degrees, "degrees")
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup nonnegative")
    if len(set(args.retention_ratios)) != len(args.retention_ratios) or any(
        not math.isfinite(ratio) or not 0 < ratio <= 1
        for ratio in args.retention_ratios
    ):
        raise ValueError("Retention ratios must be unique finite numbers in (0, 1]")
    suite_path = args.suite.resolve(strict=True)
    suite, manifests = load_suite(suite_path)
    binaries = {
        operator: (args.build_dir / operator).resolve(strict=True)
        for operator in {workload["operator"] for workload in suite["workloads"]}
    }
    settings = dict(
        ks=args.ks,
        degrees=args.degrees,
        blocks=list(BLOCKS),
        items=list(ITEMS),
        warmup=args.warmup,
        repeat=args.repeat,
        retention_ratios=args.retention_ratios,
        binaries={operator: sha256_file(path) for operator, path in binaries.items()},
        code_sha256={
            name: sha256_file(SCRIPTS / name)
            for name in (
                "run-application-scaling.py",
                "run-applications.py",
                "application_inputs.py",
            )
        },
        hostname=platform.node(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=gpu_identity(),
    )
    index = dict(
        version=1,
        suite=str(suite_path),
        suite_sha256=sha256_file(suite_path),
        settings=settings,
        settings_sha256=object_hash(settings),
        workloads=suite["workloads"],
        runs=make_runs(
            suite["workloads"], manifests, args.ks, args.degrees, args.retention_ratios
        ),
    )
    if args.dry_run:
        print(json.dumps(index, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another runner is already using this output directory"
            ) from error
        index_path = args.output_dir / "index.json"
        if index_path.exists():
            if not args.resume:
                raise ValueError(
                    "Output index exists; use --resume or choose another directory"
                )
            previous = json.loads(index_path.read_text())
            for name in ("version", "suite_sha256", "settings_sha256", "workloads"):
                if previous.get(name) != index[name]:
                    raise ValueError(f"Cannot resume: {name} differs")
            if len(previous.get("runs", [])) != len(index["runs"]):
                raise ValueError("Cannot resume: run plan differs")
            for expected, existing in zip(index["runs"], previous["runs"]):
                for name in (
                    "workload_id",
                    "mode",
                    "requested_retention",
                    "ks",
                    "csv",
                    "log",
                ):
                    if existing.get(name) != expected[name]:
                        raise ValueError(f"Cannot resume: run {name} differs")
                if existing.get("status") == "complete":
                    timing = args.output_dir / existing["csv"]
                    if not timing.is_file() or sha256_file(timing) != existing.get(
                        "csv_sha256"
                    ):
                        raise ValueError(
                            f"Cannot resume: completed timing hash differs: {timing}"
                        )
                    expected.update(existing)
        write_json(index_path, index)
        workloads = {workload["id"]: workload for workload in suite["workloads"]}
        for position, record in enumerate(index["runs"], 1):
            if record["status"] == "complete":
                print(
                    f"[{position}/{len(index['runs'])}] resume {record['csv']}",
                    file=sys.stderr,
                )
                continue
            if record["status"] == "unsupported":
                print(
                    f"[{position}/{len(index['runs'])}] unsupported {record['csv']}: "
                    f"{record['reason']}",
                    file=sys.stderr,
                )
                continue
            workload = workloads[record["workload_id"]]
            print(
                f"[{position}/{len(index['runs'])}] running {record['csv']}",
                file=sys.stderr,
            )
            record.update(status="running", started_at=now())
            write_json(index_path, index)
            try:
                ensure_gpu_idle(settings["gpu"])
                result = run_workload(
                    record,
                    workload,
                    manifests[workload["id"]][0],
                    binaries[workload["operator"]],
                    args.output_dir,
                    settings,
                )
                record.update(status="complete", **result)
            except (
                OSError,
                ValueError,
                subprocess.CalledProcessError,
                KeyboardInterrupt,
            ) as error:
                record.update(status="failed", reason=str(error), failed_at=now())
                write_json(index_path, index)
                raise
            write_json(index_path, index)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument(
        "--build-dir", type=Path, default=SCRIPTS.parent / "build-release"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument(
        "--retention-ratios",
        type=float,
        nargs="*",
        default=[],
        help="Optional gradient ratios (nearest positive k, half up); default: none",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the run index without running GPU work",
    )
    args = parser.parse_args()
    try:
        return execute(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
