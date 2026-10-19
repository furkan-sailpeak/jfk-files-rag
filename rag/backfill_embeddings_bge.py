"""
One-shot script to embed every row in jfk_pages with BAAI/bge-small-en-v1.5
into jfk_pages.embedding_bge (halfvec(384)). The app embeds queries with the
same model locally (fastembed/ONNX), so no embedding API is needed.

Run from a machine with a GPU if possible -- on an Apple M2 via PyTorch MPS it
is ~70 pages/s (~20 min for the corpus) vs ~9 pages/s on CPU. Vectors from
sentence-transformers and fastembed agree to cosine 0.99999.

    pip install sentence-transformers psycopg2-binary python-dotenv
    python rag/backfill_embeddings_bge.py            # resumes where it left off
    python rag/backfill_embeddings_bge.py --index    # build the HNSW index after

Idempotent: only processes rows where embedding_bge IS NULL.
"""
import os
import sys
import time

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
BATCH = 512
MAX_TOKENS = 512  # the model's context; longer pages are truncated

assert DATABASE_URL, "DATABASE_URL missing"


def ensure_column(conn):
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE jfk_pages ADD COLUMN IF NOT EXISTS embedding_bge halfvec({DIM})")
    conn.commit()


def build_index(conn):
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '128MB'")
        # Supabase's /dev/shm is small; a parallel build's shared segment
        # overflows it (DiskFull at 512MB). Serial build fits.
        cur.execute("SET max_parallel_maintenance_workers = 0")
        cur.execute("SET statement_timeout = 0")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS jfk_pages_embedding_bge_hnsw
            ON jfk_pages USING hnsw (embedding_bge halfvec_cosine_ops)
        """)
    conn.commit()
    print(f"HNSW index ready in {time.time() - t0:.0f}s")


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(MODEL, device=device)
    model.max_seq_length = MAX_TOKENS
    print(f"model={MODEL} device={device}")
    return model


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    conn = psycopg2.connect(DATABASE_URL)
    ensure_column(conn)
    if "--index" in sys.argv:
        build_index(conn)
        return

    with conn.cursor() as cur:
        cur.execute("""
            SELECT count(*) FROM jfk_pages
            WHERE embedding_bge IS NULL AND content IS NOT NULL AND length(trim(content)) > 0
        """)
        remaining = cur.fetchone()[0]
    print(f"rows to embed: {remaining}")

    model = load_model()
    done, last_id, t0 = 0, None, time.time()
    while True:
        with conn.cursor() as cur:
            # Keyset pagination: re-scanning for IS NULL from the start gets
            # slower every batch.
            cur.execute(
                """
                SELECT id, content FROM jfk_pages
                WHERE embedding_bge IS NULL AND content IS NOT NULL
                  AND length(trim(content)) > 0
                  AND (%s::bigint IS NULL OR id > %s)
                ORDER BY id LIMIT %s
                """,
                (last_id, last_id, BATCH),
            )
            rows = cur.fetchall()
        if not rows:
            break
        last_id = rows[-1][0]
        vecs = model.encode([r[1] for r in rows], batch_size=32, normalize_embeddings=True)
        with conn.cursor() as cur:
            execute_values(
                cur,
                "UPDATE jfk_pages AS t SET embedding_bge = v.emb::halfvec "
                "FROM (VALUES %s) AS v(id, emb) WHERE t.id = v.id",
                [(r[0], vec_literal(v)) for r, v in zip(rows, vecs)],
                template="(%s, %s)",
                page_size=BATCH,
            )
        conn.commit()
        done += len(rows)
        rate = done / max(time.time() - t0, 1e-3)
        eta = (remaining - done) / max(rate, 1e-3) / 60
        print(f"  done={done}/{remaining}  {rate:.0f} rows/s  eta={eta:.0f} min", flush=True)

    print(f"Embedded {done} rows in {(time.time() - t0) / 60:.1f} min")
    conn.close()


if __name__ == "__main__":
    main()
