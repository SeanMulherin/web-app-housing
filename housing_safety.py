"""Persistent, fail-closed admission controls for paid property lookups."""

import argparse
from contextlib import closing
import hashlib
import math
import os
from pathlib import Path
import secrets
import sqlite3
import time


BUDGET_WINDOW_SECONDS = 31 * 24 * 60 * 60
# Conservative pricing ceiling verified against https://www.rentcast.io/api:
# every attempt is treated as billable at the highest published overage rate,
# with no credit for free quota. 32 days also covers a 31-day month plus DST.
# Reconcile pricing before deployment; this bounds this app's request fees only.
SPEND_WINDOW_SECONDS = 32 * 24 * 60 * 60
MONTHLY_SPEND_CAP_CENTS = 500
REQUEST_COST_CENTS = 20


class SafetyBlocked(RuntimeError):
    def __init__(self, message, status_code=503, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def initialize_database(path):
    """Explicit operator bootstrap only; never overwrite or reset an existing ledger."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('The safety database path must be absolute.')
    # Exclusive creation protects an existing allowance from accidental resets.
    with path.open('xb'):
        pass
    with closing(sqlite3.connect(str(path))) as connection:
        with connection:
            connection.executescript('''
                CREATE TABLE metadata (version INTEGER NOT NULL, visitor_salt TEXT NOT NULL);
                CREATE TABLE rentcast_attempts (created_at REAL NOT NULL);
                CREATE INDEX attempts_time ON rentcast_attempts(created_at);
                CREATE TABLE visitor_requests (visitor TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE INDEX visitors_time ON visitor_requests(created_at);
                CREATE INDEX visitor_time ON visitor_requests(visitor, created_at);
            ''')
            connection.execute('INSERT INTO metadata VALUES (1, ?)', (secrets.token_hex(32),))
    os.chmod(path, 0o600)


def _configured_integer(name, default, minimum=0):
    try:
        value = int(os.environ.get(name, str(default)))
        if value < minimum:
            raise ValueError
        return value
    except ValueError as exc:
        raise SafetyBlocked('Property lookups are unavailable because safety settings are invalid.') from exc


def configured_store():
    path = os.environ.get('HOUSING_SAFETY_DB_PATH', '')
    if not path or not Path(path).is_absolute():
        raise SafetyBlocked('Property lookups are unavailable until persistent safety storage is configured.')
    return SafetyStore(
        path,
        max_requests=_configured_integer('HOUSING_RENTCAST_MAX_REQUESTS_31D', 0),
        visitor_per_minute=_configured_integer('HOUSING_VISITOR_REQUESTS_PER_MINUTE', 3, 1),
        visitor_per_day=_configured_integer('HOUSING_VISITOR_REQUESTS_PER_DAY', 20, 1),
        global_per_minute=_configured_integer('HOUSING_GLOBAL_REQUESTS_PER_MINUTE', 60, 1),
    )


class SafetyStore:
    def __init__(self, path, max_requests=0, visitor_per_minute=3,
                 visitor_per_day=20, global_per_minute=60, clock=time.time):
        self.path = Path(path)
        self.max_requests = max_requests
        self.visitor_per_minute = visitor_per_minute
        self.visitor_per_day = visitor_per_day
        self.global_per_minute = global_per_minute
        self.clock = clock

    def _connect(self):
        # mode=rw deliberately prevents recreation when a disk or file disappears.
        if not self.path.is_absolute():
            raise SafetyBlocked('Property lookup safety storage is unavailable.')
        connection = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=5)
        try:
            rows = connection.execute('SELECT version, visitor_salt FROM metadata').fetchall()
            if len(rows) != 1 or rows[0][0] != 1 or len(rows[0][1]) != 64:
                raise SafetyBlocked('Property lookup safety storage is unavailable.')
            return connection, rows[0][1]
        except Exception:
            connection.close()
            raise

    def _transaction(self, operation):
        try:
            connection, salt = self._connect()
            with closing(connection), connection:
                connection.execute('BEGIN IMMEDIATE')
                return operation(connection, salt, self.clock())
        except SafetyBlocked:
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            # Never fall back to an empty or process-local counter.
            raise SafetyBlocked('Property lookup safety storage is unavailable. Please try again later.') from exc

    def reserve_rentcast_request(self):
        def reserve(connection, salt, now):
            spending_count, spending_oldest = connection.execute(
                'SELECT COUNT(*), MIN(created_at) FROM rentcast_attempts WHERE created_at > ?',
                (now - SPEND_WINDOW_SECONDS,),
            ).fetchone()
            if (spending_count + 1) * REQUEST_COST_CENTS > MONTHLY_SPEND_CAP_CENTS:
                retry = max(1, math.ceil(spending_oldest + SPEND_WINDOW_SECONDS - now))
                raise SafetyBlocked('The $5 monthly property lookup budget has been reached. Please try again later.', 429, retry)
            count, oldest = connection.execute(
                'SELECT COUNT(*), MIN(created_at) FROM rentcast_attempts WHERE created_at > ?',
                (now - BUDGET_WINDOW_SECONDS,),
            ).fetchone()
            if count >= self.max_requests:
                retry = max(1, math.ceil(oldest + BUDGET_WINDOW_SECONDS - now)) if oldest is not None and self.max_requests else None
                raise SafetyBlocked('The property lookup request budget has been reached. Please try again later.', 429, retry)
            connection.execute('INSERT INTO rentcast_attempts VALUES (?)', (now,))
        # Commit before the HTTP request. Failed/time-out attempts are not refunded.
        self._transaction(reserve)

    def check_visitor(self, visitor_address):
        def admit(connection, salt, now):
            visitor = hashlib.sha256((salt + ':' + str(visitor_address or 'unknown')).encode()).hexdigest()
            connection.execute('DELETE FROM visitor_requests WHERE created_at <= ?', (now - 86400,))
            windows = (
                (60, self.visitor_per_minute, visitor),
                (86400, self.visitor_per_day, visitor),
                (60, self.global_per_minute, None),
            )
            for seconds, limit, identifier in windows:
                query = 'SELECT COUNT(*), MIN(created_at) FROM visitor_requests WHERE created_at > ?'
                parameters = [now - seconds]
                if identifier is not None:
                    query += ' AND visitor = ?'
                    parameters.append(identifier)
                count, oldest = connection.execute(query, parameters).fetchone()
                if count >= limit:
                    retry = max(1, math.ceil(oldest + seconds - now))
                    raise SafetyBlocked('Too many property analysis requests. Please try again later.', 429, retry)
            connection.execute('INSERT INTO visitor_requests VALUES (?, ?)', (visitor, now))
        self._transaction(admit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize a new persistent housing safety ledger without resetting existing usage.')
    parser.add_argument('command', choices=['init'])
    parser.add_argument('path', help='Absolute database path on a persistent disk')
    arguments = parser.parse_args()
    initialize_database(arguments.path)
    print('Safety ledger initialized. Paid requests remain disabled until an allowance is configured.')
