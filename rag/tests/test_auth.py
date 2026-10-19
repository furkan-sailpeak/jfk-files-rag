"""Quota identity and the rate limiter's behaviour when its counter store fails."""
import pytest

from app import auth, config, main


@pytest.fixture
def gate_on(mocker):
    mocker.patch.object(config, "GATE_ENABLED", True)
    mocker.patch.object(config, "AUTH_CONFIGURED", True)
    mocker.patch.object(config, "BURST_PER_MIN", 100)
    mocker.patch.object(config, "ANON_DAILY_LIMIT", 3)
    auth._local_counts.clear()


def _check(xff=None, remote="10.0.0.1"):
    headers = {"X-Forwarded-For": xff} if xff else {}
    with main.app.test_request_context(headers=headers, environ_base={"REMOTE_ADDR": remote}):
        return auth.check_rate_limit(None)


def _client_ip(xff=None, remote="10.0.0.1"):
    headers = {"X-Forwarded-For": xff} if xff else {}
    with main.app.test_request_context(headers=headers, environ_base={"REMOTE_ADDR": remote}):
        return auth._client_ip()


# --- Client identity ----------------------------------------------------------
def test_client_ip_uses_the_hop_the_proxy_added(mocker):
    mocker.patch.object(config, "TRUSTED_PROXY_HOPS", 1)
    # A client-written first entry must not become the identity.
    assert _client_ip("192.0.2.77, 198.51.100.9") == "198.51.100.9"
    assert _client_ip("198.51.100.9") == "198.51.100.9"


def test_client_ip_without_proxy_uses_socket_address(mocker):
    mocker.patch.object(config, "TRUSTED_PROXY_HOPS", 0)
    assert _client_ip("192.0.2.77", remote="203.0.113.5") == "203.0.113.5"


def test_rotating_forged_xff_does_not_reset_quota(gate_on, mocker):
    counts = {}

    def bump(bucket):
        counts[bucket] = counts.get(bucket, 0) + 1
        return counts[bucket]
    mocker.patch.object(auth, "_bump", side_effect=bump)
    mocker.patch.object(config, "TRUSTED_PROXY_HOPS", 1)
    results = [_check(f"192.0.2.{i}, 198.51.100.9")[0] for i in range(6)]
    assert results == [True, True, True, False, False, False]


# --- Counter store failure ----------------------------------------------------
def test_db_failure_falls_back_to_local_limits_not_fail_open(gate_on, mocker):
    # Regression: any DB error (e.g. pool exhausted) used to allow every request.
    mocker.patch.object(auth, "_bump", side_effect=RuntimeError("connection pool exhausted"))
    results = [_check()[0] for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_db_failure_still_enforces_burst(gate_on, mocker):
    mocker.patch.object(config, "BURST_PER_MIN", 2)
    mocker.patch.object(config, "ANON_DAILY_LIMIT", 50)
    mocker.patch.object(auth, "_bump", side_effect=RuntimeError("db down"))
    assert [_check()[1] for _ in range(3)] == ["", "", "burst"]


def test_local_fallback_windows_are_independent():
    auth._local_counts.clear()
    assert auth._local_bump("m", "ip:x", "12:00") == 1
    assert auth._local_bump("d", "ip:x", "day1") == 1
    assert auth._local_bump("m", "ip:x", "12:00") == 2
    assert auth._local_bump("m", "ip:x", "12:01") == 1   # new minute resets minute only
    assert auth._local_bump("d", "ip:x", "day1") == 2


def test_gate_off_allows_without_touching_db(mocker):
    mocker.patch.object(config, "GATE_ENABLED", False)
    bump = mocker.patch.object(auth, "_bump")
    assert _check()[0] is True
    bump.assert_not_called()
