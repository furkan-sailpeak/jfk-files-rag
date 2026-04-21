"""
One-shot script to embed every row in jfk_pages.

Idempotent: only processes rows where embedding IS NULL, so it can be
re-run if interrupted. Skips empty/whitespace content.
"""
import os
import sys
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
DIM = 1536
BATCH = 100
MAX_CHARS = 30000  # ~7500 tokens, safely under 8192 limit

assert DATABASE_URL and OPENAI_API_KEY, "env missing"
oai = OpenAI(api_key=OPENAI_API_KEY)


def vec_literal(v):
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    total_done = 0
    t0 = time.time()

    while True:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content FROM jfk_pages
                WHERE embedding IS NULL
                  AND content IS NOT NULL
                  AND length(trim(content)) > 0
                ORDER BY id
                LIMIT %s
            """, (BATCH,))
            rows = cur.fetchall()

        if not rows:
            break

        batch = [(rid, (c[:MAX_CHARS] if c else ""))
                 for rid, c in rows if c and c.strip()]
        if not batch:
            # nothing substantive in this page — shouldn't happen given filter,
            # but guard against infinite loops.
            break

        total_done += embed_and_update(conn, batch)
        rate = total_done / max(time.time() - t0, 1e-3)
        print(f"  done={total_done}  rate={rate:.0f} rows/s  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nTotal embedded: {total_done} in {time.time()-t0:.1f}s")
    conn.close()


def embed_and_update(conn, batch, retries=5):
    ids = [b[0] for b in batch]
    texts = [b[1] for b in batch]

    for attempt in range(retries):
        try:
            resp = oai.embeddings.create(model=MODEL, input=texts)
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"  embed error ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    else:
        print(f"  GIVING UP on batch of {len(batch)} rows starting id={ids[0]}",
              file=sys.stderr)
        return 0

    rows = [(rid, vec_literal(d.embedding))
            for rid, d in zip(ids, resp.data)]

    with conn.cursor() as wcur:
        execute_values(
            wcur,
            "UPDATE jfk_pages AS t SET embedding = v.emb::vector "
            "FROM (VALUES %s) AS v(id, emb) WHERE t.id = v.id",
            rows,
            template="(%s, %s)",
        )
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    main()
