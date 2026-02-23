import pandas as pd
import duckdb
import glob
import os
from tqdm import tqdm

# Paths
CATEGORIZATION_CSV = "database/jfk_categorization_combined.csv"
DOCUMENT_INDEX_CSV = "database/jfk-document-index.csv"
OCR_OUTPUT_DIR = "ocr_output"
DATABASE_PATH = "database/jfk_files.db"

def create_database():
    print("Loading categorization data...")
    df = pd.read_csv(CATEGORIZATION_CSV)
    
    print("Loading document index...")
    index_df = pd.read_csv(DOCUMENT_INDEX_CSV)
    
    # Create file_id (filename without extension)
    df['file_id'] = df['filename'].str.replace('.pdf', '', regex=False)
    index_df['file_id'] = index_df['filename'].str.replace('.pdf', '', regex=False)
    
    # Merge with index to get number_of_pages
    print("Merging with document index...")
    # index_df might have duplicate file_ids if filenames were slightly different but mapped to same ID? 
    # Actually filenames in index_df should be unique.
    df = df.merge(index_df[['file_id', 'number_of_pages']], on='file_id', how='left')
    
    # Load OCR content
    print("Loading OCR content...")
    ocr_files = glob.glob(os.path.join(OCR_OUTPUT_DIR, "*.txt"))
    ocr_map = {}
    
    for fpath in tqdm(ocr_files, desc="Reading TXT files"):
        file_id = os.path.basename(fpath).replace('.txt', '')
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Split by "--- PAGE BREAK ---"
            pages = [p.strip() for p in content.split("--- PAGE BREAK ---")]
            ocr_map[file_id] = pages

    # Function to get content for a specific page
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
    
    # Reorder columns as requested: file_id, number_of_pages, page_number, then others, then content
    cols = ['file_id', 'number_of_pages', 'page_number']
    other_cols = [c for c in df.columns if c not in cols and c != 'content']
    final_cols = cols + other_cols + ['content']
    df = df[final_cols]
    
    # Create DuckDB connection
    print(f"Creating DuckDB database at {DATABASE_PATH}...")
    conn = duckdb.connect(DATABASE_PATH)
    
    # Write to DuckDB
    # If table exists, drop it
    conn.execute("DROP TABLE IF EXISTS jfk_pages")
    conn.execute("CREATE TABLE jfk_pages AS SELECT * FROM df")
    
    print("Database created successfully.")
    
    # Check counts
    count = conn.execute("SELECT count(*) FROM jfk_pages").fetchone()[0]
    print(f"Total rows in jfk_pages: {count}")
    
    # Check a few random rows
    print("\nSample rows:")
    print(conn.execute("SELECT file_id, page_number, number_of_pages, content FROM jfk_pages LIMIT 5").fetchdf())
    
    conn.close()

if __name__ == "__main__":
    create_database()
