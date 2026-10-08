"""Offline proofs of the independent $5 app usage ceiling."""

import multiprocessing
import sqlite3

import pytest

from housing_safety import (
    MONTHLY_SPEND_CAP_CENTS,
    REQUEST_COST_CENTS,
    SPEND_WINDOW_SECONDS,
    SafetyBlocked,
    SafetyStore,
    configured_store,
    initialize_database,
)


DAY = 86400
START = 1_800_000_000.0


class MutableClock:
    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "safety.sqlite3"
    initialize_database(str(path))
    return path


def make_store(path, clock=None, max_requests=1_000_000):
    return SafetyStore(str(path), max_requests=max_requests,
                       clock=clock or MutableClock())


def assert_spend_blocked(store):
    with pytest.raises(SafetyBlocked) as caught:
        store.reserve_rentcast_request()
    assert caught.value.status_code == 429
    assert "$5" in str(caught.value)
    return caught.value


def test_fixed_spend_ceiling_and_reservation_cost():
    assert MONTHLY_SPEND_CAP_CENTS == 500
    assert REQUEST_COST_CENTS == 20
    assert SPEND_WINDOW_SECONDS == 32 * DAY


def test_25_attempts_accepted_and_26th_blocked(ledger):
    store = make_store(ledger)
    for _ in range(25):
        store.reserve_rentcast_request()
    assert_spend_blocked(store)
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rentcast_attempts").fetchone()[0] == 25


def test_huge_environment_request_allowance_cannot_bypass_spend_ceiling(ledger, monkeypatch):
    monkeypatch.setenv("HOUSING_SAFETY_DB_PATH", str(ledger))
    monkeypatch.setenv("HOUSING_RENTCAST_MAX_REQUESTS_31D", "1000000000")
    for _ in range(25):
        configured_store().reserve_rentcast_request()
    assert_spend_blocked(configured_store())


def test_new_store_instances_preserve_spend_reservations(ledger):
    for _ in range(25):
        make_store(ledger).reserve_rentcast_request()
    assert_spend_blocked(make_store(ledger, max_requests=2_000_000))


def test_31_day_expiry_does_not_reset_32_day_spend_limit(ledger):
    clock = MutableClock()
    store = make_store(ledger, clock)
    for _ in range(25):
        store.reserve_rentcast_request()
    clock.now += 31 * DAY + 1
    error = assert_spend_blocked(store)
    assert 0 < error.retry_after <= DAY
    clock.now = START + 32 * DAY - 1
    error = assert_spend_blocked(make_store(ledger, clock))
    assert error.retry_after == 1
    clock.now += 2
    make_store(ledger, clock).reserve_rentcast_request()


def test_existing_old_reservations_count_without_migration(ledger):
    # Existing records have only a timestamp, as in the prior ledger schema.
    # They are outside the original 31-day cap but inside the money window.
    with sqlite3.connect(ledger) as connection:
        connection.executemany("INSERT INTO rentcast_attempts VALUES (?)",
                               [(START - 31 * DAY - 60,)] * 25)
    assert_spend_blocked(make_store(ledger))
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rentcast_attempts").fetchone()[0] == 25


def test_original_smaller_request_allowance_still_tightens_cap(ledger):
    store = make_store(ledger, max_requests=3)
    for _ in range(3):
        store.reserve_rentcast_request()
    with pytest.raises(SafetyBlocked) as caught:
        store.reserve_rentcast_request()
    assert caught.value.status_code == 429
    # Increasing the optional cap keeps the existing three money reservations.
    store = make_store(ledger)
    for _ in range(22):
        store.reserve_rentcast_request()
    assert_spend_blocked(store)


def test_default_zero_allowance_remains_disabled(ledger):
    store = SafetyStore(str(ledger))
    with pytest.raises(SafetyBlocked) as caught:
        store.reserve_rentcast_request()
    assert caught.value.status_code == 429
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rentcast_attempts").fetchone()[0] == 0


def _process_attempts(path, gate, results):
    store = make_store(path)
    gate.wait(15)
    successes, blocked, errors = 0, 0, []
    for _ in range(10):
        try:
            store.reserve_rentcast_request()
            successes += 1
        except SafetyBlocked as exc:
            if exc.status_code == 429 and "$5" in str(exc):
                blocked += 1
            else:
                errors.append(repr(exc))
        except Exception as exc:
            errors.append(repr(exc))
    results.put((successes, blocked, errors))


def test_eight_processes_cannot_exceed_25_attempts(ledger):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    workers = [context.Process(target=_process_attempts,
                               args=(str(ledger), gate, results)) for _ in range(8)]
    try:
        for worker in workers:
            worker.start()
        gate.set()
        outcomes = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sum(row[0] for row in outcomes) == 25
        assert sum(row[1] for row in outcomes) == 55
        assert [error for row in outcomes for error in row[2]] == []
        assert_spend_blocked(make_store(ledger))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()
