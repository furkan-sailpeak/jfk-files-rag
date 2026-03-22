import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import glob
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Paths
CATEGORIZATION_CSV = "database/jfk_categorization_combined.csv"
DOCUMENT_INDEX_CSV = "database/jfk-document-index.csv"
OCR_OUTPUT_DIR = "ocr_output"

# Supabase PostgreSQL connection (session pooler)
DATABASE_URL = os.environ["DATABASE_URL"]


def split_pages(content, expected_pages=None):
    """Split OCR text into pages, handling 3 separator formats:
    1. '--- PAGE BREAK ---'
    2. 'PAGE N:\\n==============================' (backup files)
    3. '13-00000' / '00000' variants (original OCR marker)

    If expected_pages is provided, it's used to pick the best split when
    the 00000 marker is ambiguous (appears within pages too).
    """
    # Format 1: explicit PAGE BREAK
    if '--- PAGE BREAK ---' in content:
        return [p.strip() for p in content.split('--- PAGE BREAK ---')]

    # Format 2: PAGE N: + ====== (backup/processed files)
    if re.search(r'^PAGE \d+:\s*$', content, re.MULTILINE):
        parts = re.split(r'^PAGE \d+:\s*\n=+\n?', content, flags=re.MULTILINE)
        pages = [p.strip() for p in parts if p.strip()]
        # Drop the PROCESSED/TIMESTAMP header that precedes PAGE 1:
        if pages and pages[0].startswith('PROCESSED:'):
            pages = pages[1:]
        if pages:
            return pages

    # Format 3: 13-00000 / 00000 markers (with optional period)
    parts = re.split(r'\n(?:13-)?0{4,5}\.?\s*\n', content)
    if len(parts) > 1:
        pages = [p.strip() for p in parts if p.strip()]
        # Drop leading tiny header chunks (page number, "00000", file id, release line)
        while len(pages) > 1 and len(pages[0]) < 500:
            first = pages[0]
            if re.match(r'^(13-)?0{4,5}', first) or re.match(r'^\d{1,3}[₁₂₃]?\s*$', first.split('\n')[0]):
                pages = pages[1:]
            else:
                break

        # If we over-split (more parts than expected pages), merge extra parts
        # back into the nearest page — the 00000 marker appeared mid-page
        if expected_pages and len(pages) > expected_pages:
            # Merge the smallest adjacent chunks until we reach expected count
            while len(pages) > expected_pages:
                # Find the shortest chunk and merge it with its neighbor
                min_idx = min(range(len(pages)), key=lambda i: len(pages[i]))
                if min_idx == 0:
                    pages[0] = pages[0] + '\n' + pages[1]
                    pages.pop(1)
                else:
                    pages[min_idx - 1] = pages[min_idx - 1] + '\n' + pages[min_idx]
                    pages.pop(min_idx)

        return pages

    # Fallback: treat entire content as single page
    return [content.strip()]


def create_database():
    print("Loading categorization data...")
    df = pd.read_csv(CATEGORIZATION_CSV)

    print("Loading document index...")
    index_df = pd.read_csv(DOCUMENT_INDEX_CSV)

    # Create file_id (filename without extension) — index_df already has file_id
    df['file_id'] = df['filename'].str.replace('.pdf', '', regex=False)

    # Merge with index to get number_of_pages
    print("Merging with document index...")
    df = df.merge(index_df[['file_id', 'number_of_pages']], on='file_id', how='left')

    # Build expected page count lookup from categorization data
    expected_page_counts = df.groupby('file_id')['page_number'].max().to_dict()

    # Load OCR content
    print("Loading OCR content...")
    ocr_files = glob.glob(os.path.join(OCR_OUTPUT_DIR, "*.txt"))
    ocr_map = {}

    for fpath in tqdm(ocr_files, desc="Reading TXT files"):
        file_id = os.path.basename(fpath).replace('.txt', '')
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            expected = expected_page_counts.get(file_id)
            ocr_map[file_id] = split_pages(content, expected)

    def get_page_content(row):
        fid = row['file_id']
        pnum = int(row['page_number'])
        if fid in ocr_map:
            pages = ocr_map[fid]
            if 1 <= pnum <= len(pages):
                return pages[pnum - 1]
        return None

    print("Adding OCR content to dataframe...")
    df['content'] = df.apply(get_page_content, axis=1)

    # Reorder columns
    cols = ['file_id', 'number_of_pages', 'page_number']
    other_cols = [c for c in df.columns if c not in cols and c != 'content']
    final_cols = cols + other_cols + ['content']
    df = df[final_cols]

    # Connect to Supabase PostgreSQL
    print("Connecting to Supabase PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Drop and recreate table
    cur.execute("DROP TABLE IF EXISTS jfk_pages")
    cur.execute("""
        CREATE TABLE jfk_pages (
            id SERIAL PRIMARY KEY,
            file_id TEXT NOT NULL,
            number_of_pages INTEGER,
            page_number INTEGER NOT NULL,
            filename TEXT,
            document_type TEXT,
            ocr_difficulty TEXT,
            includes_handwriting BOOLEAN,
            has_shadowy_background BOOLEAN,
            document_quality TEXT,
            text_density TEXT,
            has_stamps BOOLEAN,
            has_redactions BOOLEAN,
            has_forms BOOLEAN,
            has_tables BOOLEAN,
            is_typewritten BOOLEAN,
            paper_condition TEXT,
            primary_characteristics TEXT,
            content TEXT
        )
    """)
    conn.commit()

    # Insert data in batches
    print("Inserting data into Supabase...")
    insert_cols = [c for c in final_cols]
    placeholders = ", ".join(["%s"] * len(insert_cols))
    insert_sql = f"INSERT INTO jfk_pages ({', '.join(insert_cols)}) VALUES %s"

    # Convert booleans properly
    bool_cols = [
        'includes_handwriting', 'has_shadowy_background', 'has_stamps',
        'has_redactions', 'has_forms', 'has_tables', 'is_typewritten'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).upper() == 'TRUE' else (False if str(x).upper() == 'FALSE' else None)
            )

    # Replace NaN with None for PostgreSQL
    df = df.where(pd.notnull(df), None)

    rows = [tuple(row) for row in df[insert_cols].values]

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Inserting batches"):
        batch = rows[i:i + BATCH_SIZE]
        execute_values(cur, insert_sql, batch)
        conn.commit()

    # Create indexes for common queries
    print("Creating indexes...")
    cur.execute("CREATE INDEX idx_jfk_pages_file_id ON jfk_pages (file_id)")
    cur.execute("CREATE INDEX idx_jfk_pages_document_type ON jfk_pages (document_type)")
    cur.execute("CREATE INDEX idx_jfk_pages_file_page ON jfk_pages (file_id, page_number)")
    cur.execute("CREATE INDEX idx_jfk_pages_content_fts ON jfk_pages USING GIN (to_tsvector('english', content))")
    conn.commit()

    # Verify
    cur.execute("SELECT count(*) FROM jfk_pages")
    count = cur.fetchone()[0]
    print(f"\nTotal rows in jfk_pages: {count}")

    cur.execute("SELECT file_id, page_number, number_of_pages, LEFT(content, 80) FROM jfk_pages LIMIT 5")
    print("\nSample rows:")
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()
    print("\nDone! Database migrated to Supabase PostgreSQL.")


if __name__ == "__main__":
    create_database()
