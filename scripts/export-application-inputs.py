#!/usr/bin/env python3
"""Export raw TPC-H lineitem columns for the GPU database top-N operator."""

import argparse
import json
import math
from pathlib import Path

from application_inputs import (
    COLUMN_TYPES,
    SEMANTICS,
    load_database_manifest,
    sha256_file,
)

QUERY = """SELECT CAST(l_extendedprice AS FLOAT) AS price,
                  CAST(l_discount AS FLOAT) AS discount,
                  CAST(l_quantity AS FLOAT) AS payload,
                  CAST(l_orderkey AS UBIGINT) * 8
                      + CAST(l_linenumber AS UBIGINT) AS row_id
           FROM lineitem ORDER BY l_orderkey, l_linenumber"""


def export_database(destination, scale_factor, extension_directory=None):
    import duckdb
    import numpy as np

    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale-factor must be finite and positive")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("The export destination must not already exist")
    config = {"threads": "1"}
    if extension_directory is not None:
        Path(extension_directory).mkdir(parents=True, exist_ok=True)
        config["extension_directory"] = str(Path(extension_directory).resolve())
    with duckdb.connect(config=config) as connection:
        connection.execute("INSTALL tpch")
        connection.execute("LOAD tpch")
        connection.execute("CALL dbgen(sf=?)", [scale_factor])
        extension_version = connection.execute(
            "SELECT extension_version FROM duckdb_extensions() "
            "WHERE extension_name = 'tpch'"
        ).fetchone()[0]
        arrays = connection.execute(QUERY).fetchnumpy()
    rows = len(arrays["price"])
    if not 0 < rows <= 2**31 - 1:
        raise ValueError("Generated table must have a positive int32 row count")
    # Convert exported SQL FLOAT columns, never SQL DECIMAL ranking expressions.
    for name, (dtype, _) in COLUMN_TYPES.items():
        value = arrays[name]
        if np.ma.isMaskedArray(value) and np.any(value.mask):
            raise ValueError(f"NULL values in {name}")
        arrays[name] = np.ascontiguousarray(
            value, dtype="<f4" if dtype == "float32" else "<u8"
        )
        if dtype == "float32" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite values in {name}")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = arrays["price"] * (np.float32(1) - arrays["discount"])
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite FP32 ranking scores")
    manifest = {
        "version": 1,
        "operator": "database-topn",
        "rows": rows,
        "semantics": SEMANTICS,
        "source": {
            "kind": "generated-tpch",
            "table": "lineitem",
            "duckdb_version": duckdb.__version__,
            "tpch_extension_version": extension_version,
            "numpy_version": np.__version__,
            "scale_factor": scale_factor,
            "generation": "CALL dbgen(sf=?)",
            "threads": 1,
            "seed": "DuckDB dbgen defaults; no seed override",
            "export_query": QUERY,
            "row_identity": "l_orderkey * 8 + l_linenumber",
            "payload": "l_quantity cast to FLOAT",
            "filter": "none",
            "preprocessing": (
                "Cast raw numeric columns to FP32; order by orderkey, linenumber"
            ),
        },
        "columns": {},
    }
    destination.mkdir(parents=True)
    for name, (dtype, _) in COLUMN_TYPES.items():
        filename = name + (".f32" if dtype == "float32" else ".u64")
        output = destination / filename
        arrays[name].tofile(output)
        manifest["columns"][name] = {
            "file": filename,
            "dtype": dtype,
            "shape": [rows],
            "byte_order": "little",
            "sha256": sha256_file(output),
        }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_database_manifest(manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", choices=["database-topn"])
    parser.add_argument(
        "--output", type=Path, required=True, help="New export directory"
    )
    parser.add_argument("--scale-factor", type=float, default=0.1)
    parser.add_argument(
        "--extension-directory", type=Path, help="Optional DuckDB extension cache"
    )
    args = parser.parse_args()
    try:
        path = export_database(args.output, args.scale_factor, args.extension_directory)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"{error}\n")
    print(path)


if __name__ == "__main__":
    main()
