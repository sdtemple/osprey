#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd


DEFAULT_COLUMNS = [
    "primary_label",
    "common_name",
    "sampling_rate_hz",
    "start_seconds",
    "end_seconds",
    "filename",
    "collection",
    "latitude",
    "longitude",
    "class_name",
    "dataset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a metadata CSV into spectrogram metadata by rewriting "
            "filename as '<base>-<start>s-<end>.npz'."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/combined.split.5s.2d.csv"),
        help="Input CSV path (default: data/combined.split.5s.2d.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/combined.split.5s.2d.spec.csv"),
        help="Output CSV path (default: data/combined.split.5s.2d.spec.csv)",
    )
    parser.add_argument(
        "--filename-column",
        type=str,
        default="filename",
        help="Source filename column (default: filename)",
    )
    parser.add_argument(
        "--start-column",
        type=str,
        default="start_seconds",
        help="Start time column (default: start_seconds)",
    )
    parser.add_argument(
        "--end-column",
        type=str,
        default="end_seconds",
        help="End time column (default: end_seconds)",
    )
    parser.add_argument(
        "--drop-missing-columns",
        action="store_true",
        help=(
            "If set, keep only columns that exist in the input from the default "
            "spectrogram metadata column list."
        ),
    )
    parser.add_argument(
        "--keep-all-columns",
        action="store_true",
        help="If set, keep all columns instead of restricting to the default column list.",
    )
    return parser.parse_args()


def _base_stem(filename: str) -> str:
    # Match notebook behavior: remove final extension if present.
    parts = str(filename).split(".")
    if len(parts) <= 1:
        return str(filename)
    return "".join(parts[:-1])


def _build_spec_filename(base_name: str, start: object, end: object) -> str:
    return f"{base_name}-{start}s-{end}.npz"


def build_spectrogram_metadata(
    table: pd.DataFrame,
    filename_column: str,
    start_column: str,
    end_column: str,
    keep_all_columns: bool,
    drop_missing_columns: bool,
) -> pd.DataFrame:
    required = [filename_column, start_column, end_column]
    missing_required = [c for c in required if c not in table.columns]
    if missing_required:
        raise ValueError(
            f"Missing required column(s): {missing_required}. Available columns: {list(table.columns)}"
        )

    table = table.copy()
    base_names = table[filename_column].astype(str).apply(_base_stem)
    table[filename_column] = [
        _build_spec_filename(base, start, end)
        for base, start, end in zip(base_names, table[start_column], table[end_column])
    ]

    if keep_all_columns:
        return table

    if drop_missing_columns:
        columns_to_keep = [c for c in DEFAULT_COLUMNS if c in table.columns]
    else:
        missing_default = [c for c in DEFAULT_COLUMNS if c not in table.columns]
        if missing_default:
            raise ValueError(
                "Input is missing default output columns. "
                f"Pass --drop-missing-columns to keep only available ones. Missing: {missing_default}"
            )
        columns_to_keep = DEFAULT_COLUMNS

    return cast(pd.DataFrame, table[columns_to_keep])


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    table = pd.read_csv(args.input_csv)
    output = build_spectrogram_metadata(
        table,
        filename_column=args.filename_column,
        start_column=args.start_column,
        end_column=args.end_column,
        keep_all_columns=args.keep_all_columns,
        drop_missing_columns=args.drop_missing_columns,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    print(f"Read {len(table):,} rows from {args.input_csv}")
    print(f"Wrote {len(output):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
