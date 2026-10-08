"""Durable admissions on a free external Postgres database, without local files.

PostgREST commits the SQL transaction before returning its success response.
An ambiguous response never permits a RentCast call and is never retried.
The database owns the clock, serializes admissions, and enforces the fixed cap.
"""

import hashlib
from http.client import HTTPException
import json
import re
import uuid
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from housing_safety import SafetyBlocked


_UNAVAILABLE = 'Property lookup safety storage is unavailable. Please try again later.'
_MESSAGES = {
    'spend': 'The $5 monthly property lookup budget has been reached. Please try again later.',
    'budget': 'The property lookup request budget has been reached. Please try again later.',
    'rate': 'Too many property analysis requests. Please try again later.',
    'settings': 'Property lookups are unavailable because safety settings are invalid.',
    'storage': _UNAVAILABLE,
}


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        # Never forward a server credential to a redirected destination.
        return None


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Ambiguous database response')
        result[name] = value
    return result


class SupabaseSafetyStore:
    def __init__(self, url, key, ledger_id, max_requests=0, visitor_per_minute=3,
                 visitor_per_day=20, global_per_minute=60):
        try:
            if not re.fullmatch(r'https://[a-z0-9]{20}\.supabase\.co/?', url):
                raise ValueError
            if not isinstance(key, str) or not key or any(character.isspace() for character in key):
                raise ValueError
            if str(uuid.UUID(ledger_id)) != ledger_id:
                raise ValueError
            limits = (max_requests, visitor_per_minute, visitor_per_day, global_per_minute)
            if any(type(value) is not int or value > 2_147_483_647 for value in limits):
                raise ValueError
            if max_requests < 0 or min(limits[1:]) < 1:
                raise ValueError
        except (TypeError, ValueError, AttributeError):
            raise SafetyBlocked('Property lookups are unavailable because safety settings are invalid.') from None
        self.url = url.rstrip('/')
        self.key = key
        self.ledger_id = ledger_id
        self.max_requests = max_requests
        self.visitor_per_minute = visitor_per_minute
        self.visitor_per_day = visitor_per_day
        self.global_per_minute = global_per_minute
        self._opener = build_opener(_RejectRedirects())

    def _rpc(self, function, parameters):
        headers = {'apikey': self.key, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        # New sb_secret keys use apikey; legacy service-role JWTs also need Bearer.
        if self.key.startswith('eyJ'):
            headers['Authorization'] = 'Bearer ' + self.key
        request = Request(self.url + '/rest/v1/rpc/' + function,
                          data=json.dumps({'p_ledger_id': self.ledger_id, **parameters}).encode('utf-8'),
                          headers=headers, method='POST')
        try:
            with self._opener.open(request, timeout=5) as response:
                if response.status != 200:
                    raise ValueError
                body = response.read(65537)
                if len(body) > 65536:
                    raise ValueError
                result = json.loads(body, object_pairs_hook=_unique_json_object)
            if not isinstance(result, dict):
                raise ValueError
            if type(result.get('version')) is not int or result['version'] != 1:
                raise ValueError
            if result.get('ledger_id') != self.ledger_id or type(result.get('allowed')) is not bool:
                raise ValueError
            if 'retry_after' not in result:
                raise ValueError
            retry = result['retry_after']
            if retry is not None and (type(retry) is not int or not 1 <= retry <= 32 * 86400):
                raise ValueError
            reason = result.get('reason')
            if result['allowed']:
                if reason != 'ok' or retry is not None:
                    raise ValueError
                return result
            if reason not in _MESSAGES:
                raise ValueError
        except (URLError, OSError, HTTPException, ValueError, TypeError, KeyError):
            # Do not expose remote error bodies, headers, URLs, or secrets.
            raise SafetyBlocked(_UNAVAILABLE) from None
        status_code = 429 if reason in ('spend', 'budget', 'rate') else 503
        raise SafetyBlocked(_MESSAGES[reason], status_code, retry)

    def reserve_rentcast_request(self):
        self._rpc('housing_safety_reserve', {'p_max_requests': self.max_requests})

    def check_visitor(self, visitor_address):
        # No raw visitor addresses leave the backend. Key rotation changes only
        # visitor rate buckets; the global budget remains in the same ledger.
        visitor = hashlib.sha256((self.key + ':' + str(visitor_address or 'unknown')).encode()).hexdigest()
        self._rpc('housing_safety_admit', {
            'p_visitor': visitor,
            'p_visitor_per_minute': self.visitor_per_minute,
            'p_visitor_per_day': self.visitor_per_day,
            'p_global_per_minute': self.global_per_minute,
        })

    def status(self):
        result = self._rpc('housing_safety_status', {})
        for field in ('attempt_count32d', 'attempt_count31d'):
            if type(result.get(field)) is not int or result[field] < 0:
                raise SafetyBlocked(_UNAVAILABLE)
        if result['attempt_count31d'] > result['attempt_count32d']:
            raise SafetyBlocked(_UNAVAILABLE)
        return result
