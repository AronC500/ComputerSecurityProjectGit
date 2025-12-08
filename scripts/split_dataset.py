#!/usr/bin/env python3
"""
Split dataset into train/test sets using the Zenodo reference split

Usage:
    python split_dataset.py --orgs-dir <orgs_dir> --labels <labels.csv>
                            --train-list <train_files.txt> --test-list <test_files.txt>
                            --output-dir <output_dir>
"""

import os
import sys
import csv
import argparse
import shutil
from pathlib import Path


def load_file_list(filepath):
    """Load list of filenames from text file"""
    with open(filepath, "r") as f:
        return set(line.strip() for line in f if line.strip())


def load_labels(labels_csv):
    """Load labels from CSV into dict"""
    labels = {}
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["Id"]] = int(row["Label"])
    return labels


def split_dataset(orgs_dir, labels_csv, train_list_file, test_list_file, output_dir):
    """Split dataset into train/test following Zenodo reference split"""

    print("=" * 80)
    print("SPLITTING DATASET (USING ZENODO REFERENCE SPLIT)")
    print("=" * 80)

    # Load file lists
    print(f"\n[1/6] Loading train file list from: {train_list_file}")
    train_files = load_file_list(train_list_file)
    print(f"      Train files: {len(train_files)}")

    print(f"\n[2/6] Loading test file list from: {test_list_file}")
    test_files = load_file_list(test_list_file)
    print(f"      Test files: {len(test_files)}")

    # Load labels
    print(f"\n[3/6] Loading labels from: {labels_csv}")
    all_labels = load_labels(labels_csv)
    print(f"      Total labels: {len(all_labels)}")

    # Check for overlap
    overlap = train_files & test_files
    if overlap:
        print(f"      WARNING: {len(overlap)} files in both train and test!")

    # Create output directories
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    train_org_dir = train_dir / "org"
    test_org_dir = test_dir / "org"

    for d in [train_org_dir, test_org_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/6] Copying ORG files...")
    orgs_path = Path(orgs_dir)

    train_copied = 0
    test_copied = 0
    train_labels = []
    test_labels = []

    # Get all ORG files
    org_files = {f.name: f for f in orgs_path.iterdir() if f.is_file()}
    print(f"      Available ORG files: {len(org_files)}")

    # Copy train files
    print(f"\n      Copying train files...")
    for filename in train_files:
        if filename in org_files and filename in all_labels:
            src = org_files[filename]
            dst = train_org_dir / filename
            shutil.copy2(src, dst)
            train_labels.append((filename, all_labels[filename]))
            train_copied += 1
            if train_copied % 1000 == 0:
                print(f"        Copied {train_copied} train files...")

    # Copy test files
    print(f"\n      Copying test files...")
    for filename in test_files:
        if filename in org_files and filename in all_labels:
            src = org_files[filename]
            dst = test_org_dir / filename
            shutil.copy2(src, dst)
            test_labels.append((filename, all_labels[filename]))
            test_copied += 1
            if test_copied % 1000 == 0:
                print(f"        Copied {test_copied} test files...")

    print(f"\n[5/6] Writing label files...")
    # Write train labels
    train_labels_path = train_dir / "labels.csv"
    with open(train_labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        writer.writerows(sorted(train_labels))
    print(f"      Train labels: {train_labels_path}")

    # Write test labels
    test_labels_path = test_dir / "labels.csv"
    with open(test_labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        writer.writerows(sorted(test_labels))
    print(f"      Test labels:  {test_labels_path}")

    print(f"\n[6/6] Computing statistics...")

    # Count labels
    train_mal = sum(1 for _, label in train_labels if label == 1)
    train_clean = len(train_labels) - train_mal
    test_mal = sum(1 for _, label in test_labels if label == 1)
    test_clean = len(test_labels) - test_mal

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTRAIN SET:")
    print(f"  Total files:      {len(train_labels)}")
    print(
        f"  Malicious (1):    {train_mal} ({100 * train_mal / len(train_labels):.1f}%)"
    )
    print(
        f"  Clean (0):        {train_clean} ({100 * train_clean / len(train_labels):.1f}%)"
    )
    print(f"  Directory:        {train_org_dir}")
    print(f"  Labels:           {train_labels_path}")

    print(f"\nTEST SET:")
    print(f"  Total files:      {len(test_labels)}")
    print(f"  Malicious (1):    {test_mal} ({100 * test_mal / len(test_labels):.1f}%)")
    print(
        f"  Clean (0):        {test_clean} ({100 * test_clean / len(test_labels):.1f}%)"
    )
    print(f"  Directory:        {test_org_dir}")
    print(f"  Labels:           {test_labels_path}")

    print(f"\nTOTAL: {len(train_labels) + len(test_labels)} files")
    print(
        f"Train/Test ratio: {len(train_labels)}/{len(test_labels)} "
        + f"({100 * len(train_labels) / (len(train_labels) + len(test_labels)):.1f}% / "
        + f"{100 * len(test_labels) / (len(train_labels) + len(test_labels)):.1f}%)"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset using Zenodo reference split"
    )
    parser.add_argument(
        "--orgs-dir", required=True, help="Directory containing all ORG files"
    )
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument(
        "--train-list", required=True, help="Text file with train filenames"
    )
    parser.add_argument(
        "--test-list", required=True, help="Text file with test filenames"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for train/test splits"
    )

    args = parser.parse_args()

    # Validate inputs
    for path, name in [
        (args.orgs_dir, "ORGs directory"),
        (args.labels, "Labels file"),
        (args.train_list, "Train list"),
        (args.test_list, "Test list"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    split_dataset(
        args.orgs_dir, args.labels, args.train_list, args.test_list, args.output_dir
    )


if __name__ == "__main__":
    main()
