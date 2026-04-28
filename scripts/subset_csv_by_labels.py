#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset a large CSV by keeping only rows whose label column matches a whitelist."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/combined.split.5s.2d.csv"),
        help="Input CSV file (default: data/combined.split.5s.2d.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/combined.split.5s.2d.subset.csv"),
        help="Output CSV file (default: data/combined.split.5s.2d.subset.csv)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated list of primary_label values to keep.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional text file containing one primary_label per line.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="primary_label",
        help="Name of the column to filter on (default: primary_label).",
    )
    return parser.parse_args()


def load_labels(labels_arg: str, labels_file: Path | None) -> set[str]:
    labels: set[str] = set()

    if labels_arg.strip():
        labels.update(label.strip() for label in labels_arg.split(",") if label.strip())

    if labels_file is not None:
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        with labels_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                label = line.strip()
                if label:
                    labels.add(label)

    if not labels:
        raise ValueError("Provide at least one label via --labels or --labels-file")

    return labels


def subset_csv(input_csv: Path, output_csv: Path, labels: set[str], label_column: str) -> tuple[int, int]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    kept_rows = 0
    total_rows = 0

    with input_csv.open("r", newline="", encoding="utf-8") as infile, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        if label_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{label_column}' not found in input CSV. Available columns: {reader.fieldnames}"
            )

        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            if row.get(label_column, "").strip() in labels:
                writer.writerow(row)
                kept_rows += 1

    return total_rows, kept_rows


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels, args.labels_file)
    total_rows, kept_rows = subset_csv(args.input_csv, args.output_csv, labels, args.label_column)

    print(f"Read {total_rows:,} rows from {args.input_csv}")
    print(f"Kept {kept_rows:,} rows with {args.label_column} in the provided label set")
    print(f"Wrote subset to {args.output_csv}")


if __name__ == "__main__":
    main()