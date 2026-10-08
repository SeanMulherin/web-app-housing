"""Offline regressions for durable billing and abuse protections."""

import hashlib
import multiprocessing
import sqlite3

import pytest

from housing_safety import SafetyBlocked, SafetyStore, initialize_database


DAY = 24 * 60 * 60
START = 1_800_000_000.0


class MutableClock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now


def make_store(path, clock=None, **options):
    return SafetyStore(str(path), clock=clock or MutableClock(), **options)


def assert_blocked(action, status=429):
    with pytest.raises(SafetyBlocked) as caught:
        action()
    assert caught.value.status_code == status
    return caught.value


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "safety.sqlite3"
    initialize_database(str(path))
    return path


def test_zero_budget_disables_paid_requests(ledger):
    assert_blocked(make_store(ledger).reserve_rentcast_request)


def test_budget_persists_across_store_instances_and_configuration_changes(ledger):
    make_store(ledger, max_requests=2).reserve_rentcast_request()
    make_store(ledger, max_requests=2).reserve_rentcast_request()
    assert_blocked(make_store(ledger, max_requests=2).reserve_rentcast_request)
    # Lowering the configured budget cannot hide already reserved attempts.
    assert_blocked(make_store(ledger, max_requests=1).reserve_rentcast_request)


def test_budget_uses_rolling_31_days_and_blocked_calls_do_not_extend_it(ledger):
    clock = MutableClock()
    store = make_store(ledger, clock, max_requests=1)
    store.reserve_rentcast_request()
    clock.now += 31 * DAY - 1
    error = assert_blocked(store.reserve_rentcast_request)
    assert 0 < error.retry_after <= 2
    clock.now += 2
    store.reserve_rentcast_request()
    assert_blocked(store.reserve_rentcast_request)


def test_each_reservation_commits_before_caller_proceeds(ledger):
    store = make_store(ledger, max_requests=1)
    store.reserve_rentcast_request()
    # A fresh connection sees the reservation immediately, even if the caller
    # never reaches networking or its later request fails.
    assert_blocked(make_store(ledger, max_requests=1).reserve_rentcast_request)


def _parallel_reserve(path, gate, results):
    store = make_store(path, max_requests=7)
    gate.wait(15)
    successes, blocked, errors = 0, 0, []
    for _ in range(10):
        try:
            store.reserve_rentcast_request()
            successes += 1
        except SafetyBlocked as exc:
            if exc.status_code == 429:
                blocked += 1
            else:
                errors.append(repr(exc))
        except Exception as exc:
            errors.append(repr(exc))
    results.put((successes, blocked, errors))


def test_multiple_processes_cannot_overspend_shared_budget(ledger):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    workers = [context.Process(target=_parallel_reserve,
                               args=(str(ledger), gate, results)) for _ in range(8)]
    try:
        for worker in workers:
            worker.start()
        gate.set()
        outcomes = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sum(result[0] for result in outcomes) == 7
        assert sum(result[1] for result in outcomes) == 73
        assert [error for result in outcomes for error in result[2]] == []
        assert_blocked(make_store(ledger, max_requests=7).reserve_rentcast_request)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()


@pytest.mark.parametrize("operation", ["reserve_rentcast_request", "check_visitor"])
@pytest.mark.parametrize("state", ["missing", "empty", "uninitialized", "corrupt", "directory"])
def test_unusable_ledger_fails_closed_without_replacement(tmp_path, operation, state):
    path = tmp_path / "safety.sqlite3"
    if state == "empty":
        path.write_bytes(b"")
    elif state == "uninitialized":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
    elif state == "corrupt":
        path.write_bytes(b"this is not a SQLite database")
    elif state == "directory":
        path.mkdir()
    before = hashlib.sha256(path.read_bytes()).digest() if path.is_file() else None

    def attempt():
        store = make_store(path, max_requests=10)
        if operation == "check_visitor":
            store.check_visitor("198.51.100.5")
        else:
            store.reserve_rentcast_request()

    assert_blocked(attempt, status=503)
    if state == "directory":
        assert path.is_dir()
        assert list(path.iterdir()) == []
    elif before is None:
        assert not path.exists()
    else:
        assert hashlib.sha256(path.read_bytes()).digest() == before


def test_removing_ledger_after_store_creation_fails_closed(ledger):
    store = make_store(ledger, max_requests=10)
    ledger.unlink()
    assert_blocked(store.reserve_rentcast_request, status=503)
    assert not ledger.exists()


def test_initializer_refuses_existing_file_and_preserves_it(ledger, tmp_path):
    for path in (ledger, tmp_path / "existing-empty.sqlite3"):
        if not path.exists():
            path.write_bytes(b"")
        before = path.read_bytes()
        with pytest.raises(Exception):
            initialize_database(str(path))
        assert path.read_bytes() == before


def test_visitor_minute_limit_is_shared_and_blocked_calls_do_not_extend_it(ledger):
    clock = MutableClock()
    options = dict(visitor_per_minute=1, visitor_per_day=10, global_per_minute=20)
    first = make_store(ledger, clock, **options)
    second = make_store(ledger, clock, **options)
    first.check_visitor("198.51.100.5")
    clock.now += 59
    error = assert_blocked(lambda: second.check_visitor("198.51.100.5"))
    assert 0 < error.retry_after <= 2
    # Another visitor has a separate allowance.
    second.check_visitor("198.51.100.6")
    clock.now += 2
    second.check_visitor("198.51.100.5")


def test_visitor_daily_limit_survives_minute_expiry_and_store_restart(ledger):
    clock = MutableClock()
    options = dict(visitor_per_minute=10, visitor_per_day=1, global_per_minute=20)
    make_store(ledger, clock, **options).check_visitor("198.51.100.5")
    clock.now += 61
    assert_blocked(lambda: make_store(ledger, clock, **options).check_visitor("198.51.100.5"))
    clock.now = START + DAY + 1
    make_store(ledger, clock, **options).check_visitor("198.51.100.5")


def test_global_rate_limit_is_shared_across_visitors_and_instances(ledger):
    clock = MutableClock()
    options = dict(visitor_per_minute=10, visitor_per_day=20, global_per_minute=2)
    make_store(ledger, clock, **options).check_visitor("198.51.100.5")
    make_store(ledger, clock, **options).check_visitor("198.51.100.6")
    clock.now += 59
    error = assert_blocked(lambda: make_store(ledger, clock, **options).check_visitor("198.51.100.7"))
    assert 0 < error.retry_after <= 2
    clock.now += 2
    make_store(ledger, clock, **options).check_visitor("198.51.100.7")


def test_sqlite_storage_does_not_contain_raw_visitor_addresses(ledger):
    address = "198.51.100.123"
    make_store(ledger).check_visitor(address)
    with sqlite3.connect(ledger) as connection:
        dump = "\n".join(connection.iterdump())
    assert address not in dump
    assert address.encode() not in ledger.read_bytes()
