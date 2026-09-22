"""Keep independent buffer campaigns separate, even on the same GPU host."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd


PREPROCESSORS = {"identity", "ascending", "descending"}


@dataclass
class Run:
    path: Path
    hostname: str
    jobid: int
    preprocessor: str
    layout: pd.DataFrame
    data: pd.DataFrame


def read_run(path):
    path = Path(path)
    match = re.fullmatch(r"buffer-(.+)-(\d+)\.csv", path.name)
    if match is None:
        raise ValueError(f"Invalid buffer run filename: {path}")
    data = pd.read_csv(path)
    required = {
        "algorithm",
        "generator",
        "preprocessor",
        "point_count",
        "query_count",
        "dim",
        "block_size",
        "k",
        "items_per_thread",
        "deg",
        "iteration",
        "phase",
        "time",
    }
    if set(data.columns) != required or data.empty or data.isna().any().any():
        raise ValueError(f"{path}: expected complete buffer measurements")
    preprocessors = set(data["preprocessor"])
    if len(preprocessors) != 1 or not preprocessors <= PREPROCESSORS:
        raise ValueError(f"{path}: expected one supported preprocessor")
    # Compare every measured setting, phase and repetition, including row counts.
    # Only the intended input ordering and observed timings may differ.
    columns = sorted(required - {"preprocessor", "time"})
    layout = data[columns].sort_values(columns).reset_index(drop=True)
    if layout.duplicated().any():
        raise ValueError(f"{path}: duplicate measurement")
    return Run(path, match[1], int(match[2]), preprocessors.pop(), layout, data)


def compatible(runs):
    return (
        len(runs) == 3
        and len({run.hostname for run in runs}) == 1
        and {run.preprocessor for run in runs} == PREPROCESSORS
        and all(run.layout.equals(runs[0].layout) for run in runs[1:])
    )


def group_runs(paths, registry):
    """Return validated groups and warnings; never fill a missing known group.

    Unknown campaigns must have consecutive job IDs in submission order
    identity/ascending/descending, and identical measurement layouts. This is
    deliberately conservative: exceptional campaigns belong in the registry.
    """
    database = json.loads(Path(registry).read_text())
    if database.get("version") != 1:
        raise ValueError("Unsupported multibuffer run registry version")
    runs, warnings = {}, []
    for path in sorted(map(Path, paths)):
        try:
            run = read_run(path)
        except (ValueError, OSError) as error:
            warnings.append(str(error))
            continue
        key = run.hostname, run.jobid
        if key in runs:
            raise ValueError(f"Duplicate buffer run: {key}")
        runs[key] = run

    groups, reserved, outputs = [], set(), set()
    for entry in database["groups"]:
        keys = [(entry["hostname"], job) for job in entry["jobs"]]
        output = entry["output"]
        if (
            len(keys) != 3
            or len(set(keys)) != 3
            or reserved.intersection(keys)
            or output in outputs
            or not re.fullmatch(r"multibuffer-[\w-]+", output)
        ):
            raise ValueError(
                f"Invalid or overlapping multibuffer registry entry: {entry}"
            )
        reserved.update(keys)
        outputs.add(output)
        present = [runs[key] for key in keys if key in runs]
        if not present:
            continue
        if not compatible(present):
            warnings.append(
                f"Skipping incomplete or incompatible known group: {output}"
            )
            continue
        groups.append((output, present))

    remaining = {key: run for key, run in runs.items() if key not in reserved}
    used = set()
    for key, run in sorted(remaining.items()):
        if run.preprocessor != "identity":
            continue
        keys = [(run.hostname, run.jobid + offset) for offset in range(3)]
        candidates = [remaining[key] for key in keys if key in remaining]
        if compatible(candidates) and [item.preprocessor for item in candidates] == [
            "identity",
            "ascending",
            "descending",
        ]:
            output = f"multibuffer-{run.hostname}-{run.jobid}-{run.jobid + 2}"
            if output in outputs:
                raise ValueError(f"Multibuffer output paths collide: {output}")
            outputs.add(output)
            groups.append((output, candidates))
            used.update(keys)
    for key in sorted(remaining.keys() - used):
        warnings.append(f"Skipping ungrouped run: {remaining[key].path}")
    return groups, warnings
