"""Gunicorn configuration.

The previous setup was `--workers 2` with the default *sync* worker class.
Every /api/chat request is an SSE stream held open for the full pipeline
(measured ~13s in production), and a sync worker is blocked for that entire
duration. That capped the whole service at two concurrent users.

Gevent worker: one worker handles many concurrent requests, because the time
is spent waiting on network I/O (LLM APIs, Postgres), not on CPU.
"""
import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"

worker_class = "gevent"
# Each greenlet is cheap; the real ceiling is upstream API rate limits, not
# this number.
worker_connections = int(os.getenv("WORKER_CONNECTIONS", "200"))
workers = int(os.getenv("WEB_CONCURRENCY", str(min(4, multiprocessing.cpu_count() * 2 + 1))))

# Long timeout so a slow LLM roundtrip is never killed mid-stream. With gevent
# a hung request no longer blocks other traffic, so this is safe to raise.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

# Recycle workers periodically so any slow leak can't accumulate across a
# long-running public deployment.
max_requests = 1000
max_requests_jitter = 100


def post_fork(server, worker):
    """psycopg2 is a C extension: gevent's monkey-patching does not reach it,
    so a blocking query would stall every greenlet in the worker. psycogreen
    makes psycopg2 yield to the event loop instead."""
    try:
        from psycogreen.gevent import patch_psycopg
        patch_psycopg()
        worker.log.info("psycopg2 patched for gevent")
    except Exception as e:
        worker.log.warning(f"psycogreen patch failed ({e}) — DB calls will block the worker")
