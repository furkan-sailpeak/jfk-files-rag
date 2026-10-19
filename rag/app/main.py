"""Flask app: HTTP routes only. The chat logic lives in pipeline.py.

Run locally:  python -m app.main     (from rag/, serves on :5001)
Production:   gunicorn --config gunicorn.conf.py app.main:app
"""
import os
import threading
import time

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, stream_with_context
from flask_cors import CORS

from . import config, db, llm, pipeline, prompts
from .auth import check_rate_limit, get_user
from .generation import sanitize_history

_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
app = Flask(__name__, static_folder=_FRONTEND_DIST, static_url_path="/")

if config.ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": config.ALLOWED_ORIGINS}})
    print(f"CORS locked to: {config.ALLOWED_ORIGINS}")
else:
    CORS(app)
    print("WARNING: ALLOWED_ORIGINS unset — CORS is open to all origins (dev mode).")

if config.DATABASE_URL and config.RUN_SCHEMA_BOOTSTRAP:
    db.ensure_schema()


def _rate_limited_response(user, reason):
    if reason == "burst":
        return jsonify({
            "error": "rate_limited",
            "message": "Too many requests in a short time. Please wait a moment and try again.",
        }), 429
    if user:
        return jsonify({
            "error": "daily_limit",
            "message": f"You've reached your daily limit of {config.USER_DAILY_LIMIT} requests. It resets at 00:00 UTC.",
        }), 429
    return jsonify({
        "error": "trial_exhausted",
        "message": (
            f"You've reached the limit of {config.ANON_DAILY_LIMIT} unauthenticated "
            "requests. Sign in to continue exploring the archive."
        ),
    }), 401


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get('query')
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "No query provided"}), 400
    query = query.strip()[:config.MAX_QUERY_CHARS]
    if not llm.client:
        return jsonify({"error": "LLM client not configured"}), 500

    # Free trial for anonymous visitors, quota for signed-in users.
    user = get_user()
    allowed, reason, remaining = check_rate_limit(user)
    if not allowed:
        return _rate_limited_response(user, reason)

    history = sanitize_history(data.get('history'))
    return Response(
        stream_with_context(pipeline.run(query, history)),
        mimetype='text/event-stream',
        headers={
            # Frontend reads this to show "N unauthenticated requests left"
            # before the sign-in prompt appears.
            "X-Quota-Remaining": str(remaining),
            "X-Quota-Authenticated": "1" if user else "0",
            "Cache-Control": "no-cache",
            # Stop intermediate proxies buffering the SSE stream.
            "X-Accel-Buffering": "no",
        },
    )


# --- Archive stats ----------------------------------------------------------------
# The archive is static, so these counts never change between deploys. Cached,
# and collapsed into ONE scan: this route used to run eight sequential
# aggregates over ~70k rows on every page load (8.7-19.4s in production), and
# the frontend calls it on mount, so it is every visitor's path.
_stats = {"ts": 0, "data": None}
_STATS_TTL = int(os.getenv("STATS_TTL", "3600"))
_stats_lock = threading.Lock()


def compute_full_stats():
    with db.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*),
                COUNT(DISTINCT file_id),
                COUNT(*) FILTER (WHERE content IS NOT NULL AND length(trim(content)) > 0),
                COUNT(DISTINCT file_id) FILTER (WHERE content IS NOT NULL AND length(trim(content)) > 0),
                COUNT(*) FILTER (WHERE includes_handwriting = true),
                COUNT(*) FILTER (WHERE has_stamps = true),
                COUNT(*) FILTER (WHERE has_redactions = true)
            FROM jfk_pages
        """)
        (total_pages, total_docs, pages_with_content, docs_with_content,
         handwritten_pages, stamped_pages, redacted_pages) = cur.fetchone()
        cur.execute("""
            SELECT document_type, COUNT(*) AS count FROM jfk_pages
            GROUP BY document_type ORDER BY count DESC LIMIT 5
        """)
        doc_types = cur.fetchall()

    page_pct = (pages_with_content / total_pages * 100) if total_pages else 0
    doc_pct = (docs_with_content / total_docs * 100) if total_docs else 0
    return {
        "total_pages": total_pages, "total_docs": total_docs,
        "pages_with_content": pages_with_content, "docs_with_content": docs_with_content,
        "page_content_pct": round(page_pct, 1), "doc_content_pct": round(doc_pct, 1),
        "handwritten_pages": handwritten_pages, "stamped_pages": stamped_pages,
        "redacted_pages": redacted_pages,
        "document_types": [{"type": r[0], "count": r[1]} for r in doc_types],
    }


def _stats_fresh():
    return _stats["data"] and time.time() - _stats["ts"] < _STATS_TTL


@app.route('/api/stats', methods=['GET'])
def stats_route():
    if _stats_fresh():
        return jsonify(_stats["data"])
    try:
        with _stats_lock:
            # Re-check inside the lock so a cold-cache traffic spike runs the
            # scan once, not once per concurrent visitor.
            if not _stats_fresh():
                _stats.update({"ts": time.time(), "data": compute_full_stats()})
        return jsonify(_stats["data"])
    except Exception as e:
        print(f"[stats] refresh failed: {e}")
        # Serve stale rather than erroring — the numbers are cosmetic.
        if _stats["data"]:
            return jsonify(_stats["data"])
        return jsonify({"error": "stats unavailable"}), 503


# --- Other routes -----------------------------------------------------------------
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Toolbar actions on an answer: list the names in it, or summarise it."""
    data = request.get_json(silent=True) or {}
    action, text = data.get('action'), data.get('text')
    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400
    # Unbounded text straight into an LLM call was the softest cost-drain
    # target on the API — cap it and rate-limit it like /api/chat.
    text = text[:config.MAX_ANALYZE_CHARS]
    if not llm.client:
        return jsonify({"error": "LLM client not configured"}), 500
    allowed, _, _ = check_rate_limit(get_user())
    if not allowed:
        return jsonify({"error": "rate_limited", "message": "Rate limit reached. Please wait a moment or sign in."}), 429
    templates = {"names": "analyze-names.txt", "summarize": "analyze-summarize.txt"}
    if action not in templates:
        return jsonify({"error": "Invalid action"}), 400
    try:
        result = llm.complete([{"role": "user", "content": prompts.render(templates[action], text=text)}], temperature=0.3)
        return jsonify({"result": result})
    except Exception as e:
        # Provider/driver messages carry org ids, quota numbers, DB hosts and
        # users: log them, never return them.
        print(f"[analyze] failed: {e!r}")
        return jsonify({"error": "Analysis failed. Please try again later."}), 500


@app.route('/api/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    file_id = filename.replace('.pdf', '').replace('.PDF', '')
    return redirect(f"{config.NARA_BASE_URL}/{file_id}.pdf")


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(404)
def not_found(e):
    if not request.path.startswith('/api'):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "Not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
