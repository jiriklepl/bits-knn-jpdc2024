"""Version 1 database operator input format (no CUDA or DuckDB dependency)."""

import hashlib
import json
from pathlib import Path

COLUMN_TYPES = {
    "price": ("float32", 4),
    "discount": ("float32", 4),
    "payload": ("float32", 4),
    "row_id": ("uint64", 8),
}
SEMANTICS = {
    "score": "fp32(price * fp32(1 - discount))",
    "order": "descending",
    "ties": "any_cutoff_subset",
    "masking": "none",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_database_manifest(path):
    """Validate metadata, byte counts and hashes; native code checks FP32 values."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or not _is_integer(manifest.get("version"))
        or manifest["version"] != 1
    ):
        raise ValueError("Expected an application input manifest with version 1")
    if (
        manifest.get("operator") != "database-topn"
        or manifest.get("semantics") != SEMANTICS
    ):
        raise ValueError("Unsupported operator or score semantics")
    rows = manifest.get("rows")
    if not _is_integer(rows) or not 0 < rows <= 2**31 - 1:
        raise ValueError("rows must be a positive int32 candidate count")
    if not isinstance(manifest.get("source"), dict) or not manifest["source"]:
        raise ValueError("Source provenance is required")
    columns = manifest.get("columns")
    if not isinstance(columns, dict) or set(columns) != set(COLUMN_TYPES):
        raise ValueError("Expected exactly price, discount, payload and row_id columns")
    resolved = {}
    for name, (dtype, width) in COLUMN_TYPES.items():
        column = columns[name]
        if (
            not isinstance(column, dict)
            or column.get("dtype") != dtype
            or column.get("byte_order") != "little"
            or column.get("shape") != [rows]
            or any(not _is_integer(n) for n in column.get("shape", []))
        ):
            raise ValueError(f"Invalid dtype, byte order or shape for {name}")
        filename = column.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
        ):
            raise ValueError(f"Expected a relative column file for {name}")
        array = (path.parent / filename).resolve(strict=True)
        if not array.is_relative_to(path.parent) or not array.is_file():
            raise ValueError(
                f"Column file must be inside the manifest directory: {name}"
            )
        if array.stat().st_size != rows * width:
            raise ValueError(f"Wrong byte count for {name}")
        digest = column.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or sha256_file(array) != digest
        ):
            raise ValueError(f"SHA-256 mismatch for {name}")
        resolved[name] = array
    return manifest, resolved, hashlib.sha256(raw).hexdigest()


def _is_integer(value):
    # JSON booleans must not pass as Python integer subclasses.
    return isinstance(value, int) and not isinstance(value, bool)
