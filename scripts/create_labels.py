#!/usr/bin/env python3
"""
Create labels.csv mapping ORG filenames to labels (0=benign, 1=malicious)

Usage:
    python create_labels.py --orgs-dir <orgs_dir> --malware-zip <malware.zip> --clean-zip <clean.zip> --output <labels.csv>
"""

import os
import sys
import csv
import argparse
import zipfile
from pathlib import Path


def get_malware_filenames(zip_path):
    """Extract malware filenames from zip (SHA1 hash names, no extension)"""
    filenames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Skip directory entries and log.txt
            if name.endswith("/") or "log.txt" in name:
                continue
            # Extract basename (SHA1 hash)
            basename = os.path.basename(name)
            if basename:  # Skip empty
                filenames.append(basename)
    return filenames


def get_clean_filenames(zip_path):
    """Extract clean filenames from zip (remove .pdf extension)"""
    filenames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Skip directory entries
            if name.endswith("/"):
                continue
            # Extract basename and remove .pdf extension
            basename = os.path.basename(name)
            if basename.endswith(".pdf"):
                basename_no_ext = basename[:-4]  # Remove .pdf
                filenames.append(basename_no_ext)
    return filenames


def get_org_filenames(orgs_dir):
    """Get all ORG filenames from directory"""
    orgs_path = Path(orgs_dir)
    if not orgs_path.exists():
        raise FileNotFoundError(f"ORGs directory not found: {orgs_dir}")

    return [f.name for f in orgs_path.iterdir() if f.is_file()]


def create_labels_csv(orgs_dir, malware_zip, clean_zip, output_path):
    """Create labels.csv mapping ORG filenames to labels"""

    print("=" * 80)
    print("CREATING LABELS.CSV")
    print("=" * 80)

    # Get filenames from zips
    print(f"\n[1/4] Extracting malware filenames from: {malware_zip}")
    malware_files = set(get_malware_filenames(malware_zip))
    print(f"      Found {len(malware_files)} malware files")

    print(f"\n[2/4] Extracting clean filenames from: {clean_zip}")
    clean_files = set(get_clean_filenames(clean_zip))
    print(f"      Found {len(clean_files)} clean files")

    # Get ORG filenames
    print(f"\n[3/4] Reading ORG filenames from: {orgs_dir}")
    org_files = get_org_filenames(orgs_dir)
    print(f"      Found {len(org_files)} ORG files")

    # Create labels mapping
    print(f"\n[4/4] Creating labels mapping...")
    labels = []
    malware_count = 0
    clean_count = 0
    unknown_count = 0

    for org_file in sorted(org_files):
        if org_file in malware_files:
            labels.append((org_file, 1))  # Malicious
            malware_count += 1
        elif org_file in clean_files:
            labels.append((org_file, 0))  # Benign
            clean_count += 1
        else:
            # Unknown - could be from errors in processing
            print(f"      WARNING: Unknown file: {org_file}")
            unknown_count += 1

    # Write to CSV
    print(f"\n[OUTPUT] Writing labels to: {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])  # Header
        writer.writerows(labels)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total ORG files:     {len(org_files)}")
    print(f"Labeled as malware:  {malware_count} (label=1)")
    print(f"Labeled as clean:    {clean_count} (label=0)")
    print(f"Unknown/Skipped:     {unknown_count}")
    print(f"Total labeled:       {len(labels)}")
    print(f"\nLabels written to:   {output_path}")
    print("=" * 80)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Create labels.csv for PDF ORG dataset"
    )
    parser.add_argument(
        "--orgs-dir", required=True, help="Directory containing ORG files"
    )
    parser.add_argument(
        "--malware-zip", required=True, help="Path to malware PDF zip file"
    )
    parser.add_argument("--clean-zip", required=True, help="Path to clean PDF zip file")
    parser.add_argument("--output", required=True, help="Output labels.csv path")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.malware_zip):
        print(f"ERROR: Malware zip not found: {args.malware_zip}")
        sys.exit(1)
    if not os.path.exists(args.clean_zip):
        print(f"ERROR: Clean zip not found: {args.clean_zip}")
        sys.exit(1)
    if not os.path.exists(args.orgs_dir):
        print(f"ERROR: ORGs directory not found: {args.orgs_dir}")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Create labels
    create_labels_csv(args.orgs_dir, args.malware_zip, args.clean_zip, args.output)


if __name__ == "__main__":
    main()
