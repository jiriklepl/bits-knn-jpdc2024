#!/usr/bin/env python3
"""Validate an application input manifest and run the native GPU operator."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from application_inputs import load_database_manifest, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--binary", type=Path, required=True, help="Path to database-topn"
    )
    parser.add_argument("--k", default="32")
    parser.add_argument(
        "--backends", default="bits-sq,air-topk,grid-select,block-select"
    )
    parser.add_argument("--degree", default="32")
    parser.add_argument("--bits-block-size", default="512")
    parser.add_argument(
        "--items-per-thread", help="Override BITS batches (default: bits=7, bits-sq=4)"
    )
    parser.add_argument("--repeat", default="20")
    parser.add_argument("--warmup", default="3")
    parser.add_argument(
        "--output", type=Path, help="Optional CSV of final verified projected rows"
    )
    args = parser.parse_args()
    try:
        manifest, columns, digest = load_database_manifest(args.manifest)
        command = [
            str(args.binary.resolve(strict=True)),
            "--prices",
            str(columns["price"]),
            "--discounts",
            str(columns["discount"]),
            "--payload",
            str(columns["payload"]),
            "--row-ids",
            str(columns["row_id"]),
            "--rows",
            str(manifest["rows"]),
            "--dataset-id",
            digest,
            "-k",
            args.k,
            "--backends",
            args.backends,
            "--degree",
            args.degree,
            "--bits-block-size",
            args.bits_block_size,
            "--repeat",
            args.repeat,
            "--warmup",
            args.warmup,
        ]
        if args.items_per_thread is not None:
            command += ["--items-per-thread", args.items_per_thread]
        if args.output is not None:
            if args.output.exists():
                raise ValueError("Output file already exists; choose a new file")
            command += ["--output", str(args.output.resolve())]
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest.resolve()),
                    "dataset_id": digest,
                    "source": manifest["source"],
                    "binary_sha256": sha256_file(args.binary),
                    "command": command,
                }
            ),
            file=sys.stderr,
        )
        return subprocess.run(command, check=False).returncode
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
