#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic label mappings for one-hot encoding from taxonomy primary_label values."
    )
    parser.add_argument(
        "--taxonomy-csv",
        type=Path,
        default=Path("taxonomy.csv"),
        help="Path to taxonomy CSV containing a primary_label column",
    )
    parser.add_argument(
        "--extra-labels",
        type=str,
        default="0,1,2,3",
        help="Comma-separated labels to force-include (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("onehot_labels.csv"),
        help="Output CSV mapping file (index,primary_label)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("onehot_labels.json"),
        help="Output JSON with label_to_index and index_to_label",
    )
    return parser.parse_args()


def is_int_string(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def sort_key(label: str) -> tuple[int, int | str]:
    # Numeric labels first in integer order, then alphanumeric labels.
    if is_int_string(label):
        return (0, int(label))
    return (1, label)


def read_primary_labels(taxonomy_csv: Path) -> set[str]:
    if not taxonomy_csv.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_csv}")

    labels: set[str] = set()
    with taxonomy_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "primary_label" not in (reader.fieldnames or []):
            raise ValueError("taxonomy CSV must contain a primary_label column")
        for row in reader:
            label = (row.get("primary_label") or "").strip()
            if label:
                labels.add(label)
    return labels


def write_outputs(labels: list[str], output_csv: Path, output_json: Path) -> None:
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "primary_label"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label])

    payload = {
        "num_classes": len(labels),
        "index_to_label": labels,
        "label_to_index": label_to_index,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()

    labels = read_primary_labels(args.taxonomy_csv)
    forced = [x.strip() for x in args.extra_labels.split(",") if x.strip()]
    labels.update(forced)

    ordered_labels = sorted(labels, key=sort_key)
    write_outputs(ordered_labels, args.output_csv, args.output_json)

    print(f"Wrote {len(ordered_labels)} classes to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()