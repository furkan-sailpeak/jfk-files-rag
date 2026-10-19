"""Supabase JWT verification and the per-user / per-IP rate limiter."""
import threading
import time

import jwt
from flask import request

from . import config, db

if not config.AUTH_CONFIGURED:
    print("WARNING: neither SUPABASE_URL nor SUPABASE_JWT_SECRET set — "
          "authentication disabled; the sign-in quota gate is not enforced.")

_jwks_client = None
_jwks_lock = threading.Lock()


def _get_jwks_client():
    """Lazily build a JWKS client. PyJWT caches the fetched keys, so this is
    one network call per process, not per request."""
    global _jwks_client
    if _jwks_client is not None or not config.SUPABASE_URL:
        return _jwks_client
    with _jwks_lock:
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(
                f"{config.SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                cache_keys=True,
            )
    return _jwks_client


def get_user():
    """Decode a Supabase access token from the Authorization header.
    Returns {"id", "email"} for a valid token, else None (anonymous)."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[7:].strip()
    if not token:
        return None
    try:
        # The header tells us which scheme signed this token; pick the
        # matching verification path rather than guessing.
        alg = jwt.get_unverified_header(token).get("alg", "")
        if alg.startswith("HS"):
            if not config.SUPABASE_JWT_SECRET:
                return None
            key, algorithms = config.SUPABASE_JWT_SECRET, ["HS256"]
        else:
            jwks = _get_jwks_client()
            if not jwks:
                return None
            key = jwks.get_signing_key_from_jwt(token).key
            algorithms = ["ES256", "RS256"]
        payload = jwt.decode(token, key, algorithms=algorithms, audience="authenticated")
        return {"id": payload.get("sub"), "email": payload.get("email")}
    except Exception as e:
        print(f"[auth] rejected token: {e}")
        return None


def _client_ip():
    """Client address for anonymous quota buckets.

    Behind Railway, remote_addr is the proxy, so the address comes from
    X-Forwarded-For -- but only the entry our own proxy added. Proxies append
    to the header, so everything left of the last TRUSTED_PROXY_HOPS entries
    is client-written: trusting the first entry let anyone rotate it for
    unlimited quota, or spoof a victim's IP to exhaust theirs."""
    hops = config.TRUSTED_PROXY_HOPS
    if hops > 0:
        parts = [p.strip() for p in request.headers.get("X-Forwarded-For", "").split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.remote_addr or "unknown"


def _bump(bucket):
    """Atomically increment a counter bucket and return its new value."""
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rate_limits (bucket, count) VALUES (%s, 1)
            ON CONFLICT (bucket) DO UPDATE SET count = rate_limits.count + 1
            RETURNING count
            """,
            (bucket,),
        )
        return cur.fetchone()[0]


# Per-worker fallback counters, used only while the shared counter store is
# unreachable: {"m" | "d": (current window, {identity: count})}. A new window
# replaces the old one, so memory stays bounded by one window's identities.
_local_counts = {}
_local_lock = threading.Lock()


def _local_bump(kind, ident, window):
    with _local_lock:
        current, counts = _local_counts.get(kind, (None, {}))
        if current != window:
            counts = {}
        counts[ident] = counts.get(ident, 0) + 1
        _local_counts[kind] = (window, counts)
        return counts[ident]


def check_rate_limit(user):
    """Two windows: a per-minute burst guard and a daily quota. Buckets carry
    the window in the key, so they expire implicitly — no sweeper needed on
    the hot path.

    Returns (allowed, reason, remaining_today).

    If the shared counter store (Postgres) errors, the same limits are
    enforced with per-worker in-memory counters instead. This used to fail
    open: under concurrent load the connection pool ran out, every check
    raised, and every request was allowed with no burst or daily limit.
    """
    if not config.GATE_ENABLED:
        return True, "", config.USER_DAILY_LIMIT

    now = time.gmtime()
    day = time.strftime("%Y-%m-%d", now)
    minute = time.strftime("%Y-%m-%dT%H:%M", now)

    if user:
        ident, daily_cap = f"user:{user['id']}", config.USER_DAILY_LIMIT
    else:
        # When auth isn't configured there is no sign-in to upgrade to, so
        # applying the small unauthenticated allowance would strand the user
        # at a dead end. Fall back to the full quota; the burst guard still
        # applies.
        ident = f"ip:{_client_ip()}"
        daily_cap = config.ANON_DAILY_LIMIT if config.AUTH_CONFIGURED else config.USER_DAILY_LIMIT

    try:
        burst = _bump(f"{ident}:m:{minute}")
        used = _bump(f"{ident}:d:{day}") if burst <= config.BURST_PER_MIN else None
    except Exception as e:
        print(f"[ratelimit] counter store unavailable, using per-worker limits: {e!r}")
        burst = _local_bump("m", ident, minute)
        used = _local_bump("d", ident, day) if burst <= config.BURST_PER_MIN else None

    if burst > config.BURST_PER_MIN:
        return False, "burst", 0

    if used > daily_cap:
        return False, "daily", 0
    return True, "", max(0, daily_cap - used)
