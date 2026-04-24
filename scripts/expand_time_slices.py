import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand audio metadata into time slices."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="combined.metadata.split.csv",
        help="Input metadata CSV file with dataset assignments (default: combined.metadata.split.csv)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="combined.metadata.split.timeslices.csv",
        help="Output CSV file with time slices (default: combined.metadata.split.timeslices.csv)"
    )
    parser.add_argument(
        "--slice-size-seconds",
        type=float,
        default=5.0,
        help="Duration of each time slice in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=None,
        help="Stride between slice starts in seconds (default: same as slice-size for non-overlapping slices)"
    )
    parser.add_argument(
        "--minimum-duration-seconds",
        type=float,
        default=2.,
        help="Skip files shorter than this duration in seconds (default: 0.0)"
    )
    return parser.parse_args()


def expand_row_into_slices(row: Dict, slice_size: float, stride: float, minimum_duration: float) -> List[Dict]:
    """Expand a single row into multiple rows, one per time slice.
    
    Args:
        row: Input row dict containing 'duration' field
        slice_size: Duration of each time slice in seconds
        stride: Stride between slice starts in seconds
    
    Returns:
        List of new rows, one per time slice
    """
    try:
        duration = float(row.get('duration', 0))
    except ValueError:
        # If duration is not a valid number, return empty list
        return []
    
    if duration <= 0 or duration < minimum_duration:
        return []
    
    slices = []
    epsilon = 1e-9
    
    # Generate time slices.
    # Short files get one slice from 0 to duration.
    # Longer files get overlapping or non-overlapping windows plus a final slice ending exactly at duration.
    if duration <= slice_size + epsilon:
        new_row = row.copy()
        new_row['start_seconds'] = "0.000000"
        new_row['end_seconds'] = f"{duration:.6f}"
        return [new_row]

    start_time = 0.0
    while start_time + slice_size < duration - epsilon:
        end_time = start_time + slice_size
        
        # Create a new row for this slice
        new_row = row.copy()
        new_row['start_seconds'] = f"{start_time:.6f}"
        new_row['end_seconds'] = f"{end_time:.6f}"
        slices.append(new_row)
        
        start_time += stride
    
    final_start = max(0.0, duration - slice_size)
    final_end = duration
    if not slices or final_start > float(slices[-1]['start_seconds']) + epsilon:
        new_row = row.copy()
        new_row['start_seconds'] = f"{final_start:.6f}"
        new_row['end_seconds'] = f"{final_end:.6f}"
        slices.append(new_row)
    else:
        slices[-1]['end_seconds'] = f"{final_end:.6f}"
    
    return slices


def expand_csv(
    input_csv: str,
    output_csv: str,
    slice_size: float,
    stride: float,
    minimum_duration: float
):
    """Expand CSV rows into time slices.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        slice_size: Duration of each time slice in seconds
        stride: Stride between slice starts in seconds
    """
    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slices = expand_row_into_slices(row, slice_size, stride, minimum_duration)
            rows.extend(slices)
    
    if not rows:
        print("No rows to write.")
        return
    
    # Get fieldnames from first row
    fieldnames = list(rows[0].keys())
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_csv}")


def main():
    args = parse_args()
    
    # Default stride to slice_size if not specified
    stride = args.stride_seconds if args.stride_seconds is not None else args.slice_size_seconds
    
    # Validate parameters
    if args.slice_size_seconds <= 0:
        raise ValueError("slice_size_seconds must be positive")
    if stride <= 0:
        raise ValueError("stride_seconds must be positive")
    
    print(f"Reading {args.input_csv}...")
    
    # Count input rows
    input_row_count = 0
    with open(args.input_csv, 'r') as f:
        input_row_count = sum(1 for _ in f) - 1  # Subtract header
    print(f"  {input_row_count} input rows")
    
    print(f"\nExpanding into time slices...")
    print(f"  Slice size: {args.slice_size_seconds} seconds")
    print(f"  Stride: {stride} seconds")
    print(f"  Minimum duration: {args.minimum_duration_seconds} seconds")
    
    expand_csv(args.input_csv, args.output_csv, args.slice_size_seconds, stride, args.minimum_duration_seconds)


if __name__ == '__main__':
    main()
