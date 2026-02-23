"""
Compare txt files between ocr_output/ and jfk_2025_text/ using jfk-document-index.csv as reference.

Reports:
1. Files in backup (jfk_2025_text) missing from ocr_output — candidates to restore
2. Files in ocr_output missing from backup (jfk_2025_text)
3. Files in CSV index missing from both directories
"""

import csv
import os
from pathlib import Path

BASE = Path(__file__).parent
OCR_DIR = BASE / "ocr_output"
BACKUP_DIR = BASE / "jfk_2025_text"
CSV_PATH = BASE / "database" / "jfk-document-index.csv"

# --- Load file IDs from CSV ---
csv_ids = set()
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_ids.add(row["file_id"].strip())

# --- Scan directories ---
ocr_files = {f.removesuffix(".txt") for f in os.listdir(OCR_DIR) if f.endswith(".txt")}
backup_files = {f.removesuffix("_processed.txt") for f in os.listdir(BACKUP_DIR) if f.endswith("_processed.txt")}

# --- Comparisons ---
in_backup_not_ocr = sorted(backup_files - ocr_files)
in_ocr_not_backup = sorted(ocr_files - backup_files)
in_csv_not_ocr = sorted(csv_ids - ocr_files)
in_csv_not_backup = sorted(csv_ids - backup_files)
in_csv_not_either = sorted(csv_ids - ocr_files - backup_files)

print("=" * 70)
print("FILE COMPARISON REPORT")
print("=" * 70)

print(f"\nCSV index entries:        {len(csv_ids)}")
print(f"Files in ocr_output/:     {len(ocr_files)}")
print(f"Files in jfk_2025_text/:  {len(backup_files)}")

# 1) Restore candidates: in backup but missing from ocr_output
print(f"\n{'=' * 70}")
print(f"[1] IN BACKUP but MISSING from ocr_output — restore these ({len(in_backup_not_ocr)})")
print("=" * 70)
for fid in in_backup_not_ocr:
    print(f"  {fid}")

# 2) In ocr_output but not in backup
print(f"\n{'=' * 70}")
print(f"[2] IN ocr_output but MISSING from backup ({len(in_ocr_not_backup)})")
print("=" * 70)
for fid in in_ocr_not_backup:
    print(f"  {fid}")

# 3) In CSV but missing from ocr_output
print(f"\n{'=' * 70}")
print(f"[3] IN CSV index but MISSING from ocr_output ({len(in_csv_not_ocr)})")
print("=" * 70)
for fid in in_csv_not_ocr:
    src = "backup" if fid in backup_files else "NOWHERE"
    print(f"  {fid}  [available in: {src}]")

# 4) In CSV but missing from both
print(f"\n{'=' * 70}")
print(f"[4] IN CSV index but MISSING from BOTH directories ({len(in_csv_not_either)})")
print("=" * 70)
for fid in in_csv_not_either:
    print(f"  {fid}")

# --- Summary ---
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Files to restore from backup -> ocr_output:  {len(in_backup_not_ocr)}")
print(f"  Files only in ocr_output (no backup):        {len(in_ocr_not_backup)}")
print(f"  CSV entries missing from ocr_output:          {len(in_csv_not_ocr)}")
print(f"  CSV entries missing from both dirs:           {len(in_csv_not_either)}")
