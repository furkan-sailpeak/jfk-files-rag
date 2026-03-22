#!/usr/bin/env python3
"""
Fill missing page content in Supabase from JFK_Pages_Merged.csv.

Finds all rows in jfk_pages where content is NULL/empty,
looks up the matching content from the CSV by (file_id, page_number),
and updates the DB rows. Only UPDATEs — never deletes anything.
"""

import csv
import os
import sys
import time
import psycopg2
from dotenv import load_dotenv

CSV_PATH = "/Users/furkandemir/Desktop/Thesis/database/JFK_Pages_Merged.csv"


def progress_bar(current, total, width=40, label=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {label} |{bar}| {current:,}/{total:,} ({pct:.1%})")
    sys.stdout.flush()


def main():
    load_dotenv()
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set in environment")
        sys.exit(1)

    # ── Step 1: Query missing rows from Supabase ──
    print("=" * 60)
    print("  JFK Pages — Fill Missing Content")
    print("=" * 60)
    print()
    print("[1/4] Connecting to Supabase...")

    t0 = time.time()
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print(f"       Connected in {time.time() - t0:.1f}s")

    print("[1/4] Querying pages with missing content...")
    t0 = time.time()
    cur.execute("""
        SELECT id, file_id, page_number
        FROM jfk_pages
        WHERE content IS NULL OR content = '' OR trim(content) = ''
    """)
    missing_rows = cur.fetchall()
    elapsed = time.time() - t0
    print(f"       Found {len(missing_rows):,} pages with missing content ({elapsed:.1f}s)")

    if not missing_rows:
        print("\n       Nothing to update. All pages have content!")
        cur.close()
        conn.close()
        return

    # Build lookup: (file_id, page_number) -> db row id
    missing_lookup = {}
    for row_id, file_id, page_number in missing_rows:
        missing_lookup[(file_id, page_number)] = row_id

    missing_file_ids = set(file_id for _, file_id, _ in missing_rows)
    print(f"       Across {len(missing_file_ids):,} distinct documents")
    print()

    # ── Step 2: Read CSV and match ──
    print(f"[2/4] Reading CSV and matching content...")
    print(f"       Source: {CSV_PATH}")

    # Count CSV lines first for progress
    t0 = time.time()
    with open(CSV_PATH, "r") as f:
        total_lines = sum(1 for _ in f) - 1  # minus header
    print(f"       CSV has {total_lines:,} rows")

    updates = []  # list of (content, db_row_id)
    scanned = 0

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            scanned += 1
            if scanned % 5000 == 0:
                progress_bar(scanned, total_lines, label="Scanning")

            file_id = row[1]
            page_number = int(row[3])
            content = row[18] if len(row) > 18 else ""

            key = (file_id, page_number)
            if key in missing_lookup and content.strip():
                updates.append((content, missing_lookup[key]))

    progress_bar(total_lines, total_lines, label="Scanning")
    elapsed = time.time() - t0
    print(f"\n       Scan complete in {elapsed:.1f}s")
    print(f"       Matched: {len(updates):,} pages with content from CSV")

    not_found = len(missing_rows) - len(updates)
    if not_found > 0:
        print(f"       No content in CSV: {not_found:,} pages (truly empty)")
    print()

    if not updates:
        print("       No updates to apply.")
        cur.close()
        conn.close()
        return

    # ── Step 3: Batch update ──
    print(f"[3/4] Updating Supabase ({len(updates):,} rows)...")
    batch_size = 500
    updated = 0
    t0 = time.time()

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        cur.executemany(
            "UPDATE jfk_pages SET content = %s WHERE id = %s",
            batch,
        )
        conn.commit()
        updated += len(batch)
        progress_bar(updated, len(updates), label="Uploading")

    elapsed = time.time() - t0
    print(f"\n       Upload complete in {elapsed:.1f}s")
    print()

    # ── Step 4: Verify ──
    print("[4/4] Verifying...")
    cur.execute("SELECT COUNT(*) FROM jfk_pages")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*)
        FROM jfk_pages
        WHERE content IS NULL OR content = '' OR trim(content) = ''
    """)
    remaining = cur.fetchone()[0]
    has_content = total - remaining

    print(f"       Total pages:           {total:,}")
    print(f"       Pages with content:    {has_content:,}")
    print(f"       Still missing:         {remaining:,}")
    print(f"       Coverage:              {has_content/total:.1%}")
    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
