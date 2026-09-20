"""Version 1 application input formats, using only the Python standard library."""

import hashlib
import json
import math
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
SAMPLING_SEMANTICS = {
    "selection": "largest_logits",
    "normalization": "softmax_selected_logits_over_temperature",
    "ties": "any_cutoff_subset",
    "masking": "negative_infinity",
}
GRADIENT_SEMANTICS = {
    "selection": "largest_absolute_value",
    "scope": "one_complete_parameter_tensor",
    "output": "original_signed_values",
    "ties": "any_cutoff_subset",
    "state": "stateless",
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


def _load_tensor_manifest(path, operator, semantics):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or not _is_integer(manifest.get("version"))
        or manifest["version"] != 1
    ):
        raise ValueError("Expected an application input manifest with version 1")
    if manifest.get("operator") != operator or manifest.get("semantics") != semantics:
        raise ValueError("Unsupported operator or tensor semantics")
    if not isinstance(manifest.get("source"), dict) or not manifest["source"]:
        raise ValueError("Source provenance is required")
    if operator == "token-sampling":
        batch_size = manifest.get("batch_size")
        vocabulary_size = manifest.get("vocabulary_size")
        for name, count in [
            ("batch_size", batch_size),
            ("vocabulary_size", vocabulary_size),
        ]:
            if not _is_integer(count) or not 0 < count <= 2**31 - 1:
                raise ValueError(f"{name} must be a positive int32 count")
        shape = [batch_size, vocabulary_size]
        column_name = "logits"
    else:
        elements = manifest.get("elements")
        if not _is_integer(elements) or not 0 < elements <= 2**31 - 1:
            raise ValueError("elements must be a positive int32 candidate count")
        shape = manifest.get("tensor_shape")
        if (
            not isinstance(shape, list)
            or any(not _is_integer(n) or n <= 0 for n in shape)
            or math.prod(shape) != elements
        ):
            raise ValueError("tensor_shape must describe the complete gradient tensor")
        column_name = "gradient"
    columns = manifest.get("columns")
    if not isinstance(columns, dict) or set(columns) != {column_name}:
        raise ValueError(f"Expected exactly the {column_name} column")
    column = columns[column_name]
    if (
        not isinstance(column, dict)
        or column.get("dtype") != "float32"
        or column.get("byte_order") != "little"
        or column.get("shape") != shape
        or any(not _is_integer(n) for n in column.get("shape", []))
    ):
        raise ValueError(f"Invalid dtype, byte order or shape for {column_name}")
    filename = column.get("file")
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise ValueError(f"Expected a relative column file for {column_name}")
    array = (path.parent / filename).resolve(strict=True)
    if not array.is_relative_to(path.parent) or not array.is_file():
        raise ValueError(
            f"Column file must be inside the manifest directory: {column_name}"
        )
    if array.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"Wrong byte count for {column_name}")
    digest = column.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or sha256_file(array) != digest
    ):
        raise ValueError(f"SHA-256 mismatch for {column_name}")
    return manifest, {column_name: array}, hashlib.sha256(raw).hexdigest()


def load_sampling_manifest(path):
    """Validate FP32 batch-by-vocabulary storage; native code checks values/masks."""
    return _load_tensor_manifest(path, "token-sampling", SAMPLING_SEMANTICS)


def load_gradient_manifest(path):
    """Validate one complete FP32 gradient tensor, retaining its original shape."""
    return _load_tensor_manifest(path, "gradient-compression", GRADIENT_SEMANTICS)


def load_application_manifest(path):
    """Dispatch to the strict, versioned schema for a supported application."""
    manifest = json.loads(Path(path).read_bytes())
    loaders = {
        "database-topn": load_database_manifest,
        "token-sampling": load_sampling_manifest,
        "gradient-compression": load_gradient_manifest,
    }
    if not isinstance(manifest, dict) or not isinstance(manifest.get("operator"), str):
        raise ValueError("Expected a supported application operator")
    loader = loaders.get(manifest["operator"])
    if loader is None:
        raise ValueError("Expected a supported application operator")
    return loader(path)
