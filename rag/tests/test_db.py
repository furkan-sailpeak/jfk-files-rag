import threading

import pytest
from psycopg2 import pool as pgpool

from app import config, db


def test_cursor_waits_then_raises_when_no_connection_frees(mocker):
    # Regression: the pool raised at once when exhausted; callers now queue
    # for up to DB_POOL_WAIT_S and only then fail.
    mocker.patch.object(db, "_get_pool", return_value=mocker.MagicMock())
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    mocker.patch.object(db, "_slots", slots)
    mocker.patch.object(config, "DB_POOL_WAIT_S", 0.05)
    with pytest.raises(pgpool.PoolError):
        with db.cursor():
            pass


def test_cursor_returns_slot_after_use(mocker):
    fake_pool = mocker.MagicMock()
    mocker.patch.object(db, "_get_pool", return_value=fake_pool)
    slots = threading.BoundedSemaphore(1)
    mocker.patch.object(db, "_slots", slots)
    for _ in range(3):  # would block on the 2nd use if the slot leaked
        with db.cursor():
            pass
    assert fake_pool.putconn.call_count == 3


def test_cursor_releases_slot_when_getconn_fails(mocker):
    fake_pool = mocker.MagicMock()
    fake_pool.getconn.side_effect = RuntimeError("connect failed")
    mocker.patch.object(db, "_get_pool", return_value=fake_pool)
    slots = threading.BoundedSemaphore(1)
    mocker.patch.object(db, "_slots", slots)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            with db.cursor():
                pass
