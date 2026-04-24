import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split audio metadata into train/validate/test datasets on a file-by-file basis."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="combined.metadata.csv",
        help="Input metadata CSV file (default: combined.metadata.csv)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="combined.metadata.split.csv",
        help="Output CSV file with dataset assignments (default: combined.metadata.split.csv)"
    )
    parser.add_argument(
        "--include-secondary-labels",
        action="store_true",
        help="Include rows with secondary labels; default behavior excludes rows where secondary_labels != '[]'"
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=3.,
        help="Threshold on XC rating quality (default 3)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of files for training (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of files for validation (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def filter_rows(input_csv: str, include_secondary: bool, rating_threshold: float) -> List[Dict]:
    """Read CSV and filter based on secondary_labels setting and XC rating.
    
    By default (include_secondary=False), only rows with secondary_labels == '[]' are kept.
    Files from XC collection with rating < 3 are excluded.
    """
    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            secondary_labels = row.get('secondary_labels', '[]').strip()
            
            # Filter out XC files with rating < 3
            collection = row.get('collection', '').strip()
            rating_str = row.get('rating', '').strip()
            if collection == 'XC' and rating_str:
                try:
                    rating = float(rating_str)
                    if rating < rating_threshold:
                        continue  # Skip this row
                except ValueError:
                    pass  # If rating is not a valid number, include the row
            
            if not include_secondary:
                # Default: keep only rows with no secondary labels
                if secondary_labels == '[]':
                    rows.append(row)
            else:
                # Include all rows
                rows.append(row)
    
    return rows


def group_by_primary_and_filename(rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Group rows by primary_label, then by filename.
    
    Returns:
        {primary_label: {filename: [rows]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        primary_label = row.get('primary_label', '')
        filename = row.get('filename', '')
        grouped[primary_label][filename].append(row)
    
    return grouped


def split_files_for_label(
    filenames: List[str],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random
) -> Dict[str, str]:
    """Split files for a single primary_label with constraints.
    
    Constraints:
    - At least 1 file must be in train
    - If > 2 files, at least 1 must be in test
    
    Args:
        filenames: List of unique filenames for this label
        train_ratio: Target fraction for training
        val_ratio: Target fraction for validation
        rng: Random number generator for reproducibility
    
    Returns:
        {filename: dataset} where dataset is 'train', 'validate', or 'test'
    """
    n_files = len(filenames)
    
    if n_files == 0:
        return {}
    
    # Shuffle files
    shuffled = filenames.copy()
    rng.shuffle(shuffled)
    
    # Special case: only 1 file
    if n_files == 1:
        return {shuffled[0]: 'train'}
    
    # Special case: only 2 files
    if n_files == 2:
        return {shuffled[0]: 'train', shuffled[1]: 'validate'}
    
    # General case: n_files >= 3
    # Calculate target counts
    n_train = max(1, int(n_files * train_ratio))
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val
    
    # Enforce constraint: if > 2 files, at least 1 must be in test
    if n_test == 0:
        n_test = 1
        # Reduce val first, then train if necessary
        if n_val > 0:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    
    # Assign files
    assignment = {}
    idx = 0
    
    # Assign to train
    for i in range(n_train):
        assignment[shuffled[idx]] = 'train'
        idx += 1
    
    # Assign to validate
    for i in range(n_val):
        assignment[shuffled[idx]] = 'validate'
        idx += 1
    
    # Assign to test
    for i in range(n_test):
        assignment[shuffled[idx]] = 'test'
        idx += 1
    
    return assignment


def assign_datasets(
    grouped: Dict[str, Dict[str, List[Dict]]],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> List[Dict]:
    """Assign each row to train/validate/test dataset.
    
    Splitting is done per primary_label, on a file-by-file basis.
    
    Returns:
        List of rows with 'dataset' column added
    """
    rng = random.Random(seed)
    output_rows = []
    
    for primary_label in sorted(grouped.keys()):
        files_dict = grouped[primary_label]
        filenames = list(files_dict.keys())
        
        # Split files for this label
        file_assignment = split_files_for_label(
            filenames,
            train_ratio,
            val_ratio,
            rng
        )
        
        # Assign dataset to all rows for this label
        for filename, rows_for_file in files_dict.items():
            dataset = file_assignment[filename]
            for row in rows_for_file:
                row['dataset'] = dataset
                output_rows.append(row)
    
    return output_rows


def verify_constraints(rows: List[Dict], grouped: Dict[str, Dict[str, List[Dict]]]) -> bool:
    """Verify that constraints are satisfied.
    
    Returns:
        True if all constraints are satisfied, False otherwise
    """
    all_satisfied = True
    
    label_in_train = set()
    label_in_test = set()
    label_files_in_split = defaultdict(lambda: defaultdict(set))
    
    for row in rows:
        label = row['primary_label']
        filename = row['filename']
        dataset = row['dataset']
        
        if dataset == 'train':
            label_in_train.add(label)
        if dataset == 'test':
            label_in_test.add(label)
        
        label_files_in_split[label][dataset].add(filename)
    
    # Verify constraint 1: each unique primary_label in training data
    for label in grouped.keys():
        if label not in label_in_train:
            print(f"  ✗ {label}: NOT in training data")
            all_satisfied = False
        else:
            print(f"  ✓ {label}: in training data")
    
    print()
    
    # Verify constraint 2: if > 2 files for a label, at least 1 in test
    for label in grouped.keys():
        n_files = len(grouped[label])
        if n_files > 2:
            if label not in label_in_test:
                print(f"  ✗ {label}: has {n_files} files but NOT in test data")
                all_satisfied = False
            else:
                n_test_files = len(label_files_in_split[label]['test'])
                print(f"  ✓ {label}: has {n_files} files, {n_test_files} in test")
    
    return all_satisfied


def write_output(output_csv: str, rows: List[Dict]):
    """Write rows to output CSV with dataset column, excluding author/license/rating/url."""
    if not rows:
        print("No rows to write.")
        return
    
    # Get all fieldnames from first row, ensure 'dataset' is included
    fieldnames = list(rows[0].keys())
    if 'dataset' not in fieldnames:
        fieldnames.append('dataset')
    
    # Exclude unnecessary columns
    exclude_cols = {'author', 'license', 'rating', 'url'}
    fieldnames = [col for col in fieldnames if col not in exclude_cols]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_csv}")


def main():
    args = parse_args()
    
    # Validate ratios
    if args.train_ratio < 0 or args.val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative")
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    
    print(f"Reading {args.input_csv}...")
    rows = filter_rows(args.input_csv, args.include_secondary_labels, args.rating_threshold)
    print(f"  Filtered to {len(rows)} rows")
    if args.include_secondary_labels:
        print(f"  (included rows with secondary labels)")
    else:
        print(f"  (included only rows with no secondary labels: secondary_labels == '[]')")
    print(f"  (excluded XC files with rating < {args.rating_threshold})")
    
    print("\nGrouping by primary_label and filename...")
    grouped = group_by_primary_and_filename(rows)
    print(f"  {len(grouped)} unique primary labels")
    
    print(f"\nSplitting with train_ratio={args.train_ratio}, val_ratio={args.val_ratio}...")
    output_rows = assign_datasets(grouped, args.train_ratio, args.val_ratio, args.seed)
    
    # Print statistics
    dataset_counts = Counter(row['dataset'] for row in output_rows)
    print(f"  train: {dataset_counts['train']} rows")
    print(f"  validate: {dataset_counts['validate']} rows")
    print(f"  test: {dataset_counts['test']} rows")
    
    # Verify constraints
    print("\nVerifying constraints:")
    print("  Constraint 1: Each unique primary_label appears in training data:")
    constraints_satisfied = verify_constraints(output_rows, grouped)
    
    if constraints_satisfied:
        print("\n✓ All constraints satisfied!")
    else:
        print("\n✗ Some constraints not satisfied!")
    
    print("\nWriting output...")
    write_output(args.output_csv, output_rows)


if __name__ == '__main__':
    main()
