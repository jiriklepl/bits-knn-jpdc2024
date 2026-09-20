#!/usr/bin/env python3
"""Export reproducible TPC-H columns or captured model tensors for GPU operators."""

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
    operators = parser.add_subparsers(dest="operator", required=True)
    database = operators.add_parser(
        "database-topn", help="Generate TPC-H lineitem columns"
    )
    database.add_argument(
        "--output", type=Path, required=True, help="New export directory"
    )
    database.add_argument("--scale-factor", type=float, default=0.1)
    database.add_argument(
        "--extension-directory", type=Path, help="Optional DuckDB extension cache"
    )
    for operator in ["token-sampling", "gradient-compression"]:
        model_parser = operators.add_parser(
            operator, help="Capture a real causal model on CPU"
        )
        model_parser.add_argument(
            "--output", type=Path, required=True, help="New export directory"
        )
        model_parser.add_argument("--model", default="distilbert/distilgpt2")
        model_parser.add_argument(
            "--revision",
            help="Immutable full model commit SHA; required for custom models",
        )
        model_parser.add_argument(
            "--prompts-file",
            type=Path,
            help="JSON list of prompt strings (default: eight fixed sentences)",
        )
        model_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Capture seed (independent of native sampling seed)",
        )
        model_parser.add_argument(
            "--threads", type=int, default=1, help="CPU capture threads"
        )
        model_parser.add_argument(
            "--cache-directory", type=Path, help="Optional Hugging Face model cache"
        )
        if operator == "gradient-compression":
            model_parser.add_argument(
                "--parameter",
                default="transformer.h.0.mlp.c_fc.weight",
                help="Complete named parameter to differentiate",
            )
    args = parser.parse_args()
    try:
        if args.operator == "database-topn":
            path = export_database(
                args.output, args.scale_factor, args.extension_directory
            )
        else:
            from model_application_inputs import export_model_input

            prompts = None
            if args.prompts_file is not None:
                prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
                if not isinstance(prompts, list):
                    raise ValueError(
                        "prompts must be a nonempty JSON list of nonempty strings"
                    )
            path = export_model_input(
                args.operator,
                args.output,
                model=args.model,
                revision=args.revision,
                prompts=prompts,
                parameter=getattr(args, "parameter", None),
                seed=args.seed,
                threads=args.threads,
                cache_directory=args.cache_directory,
            )
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(path)


if __name__ == "__main__":
    main()
