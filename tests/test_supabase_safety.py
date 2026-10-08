"""Offline checks of the durable admission boundary; never call real services."""

import hashlib
import io
import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler

import pytest

import housing_safety
import supabase_safety
from housing_safety import SafetyBlocked
from rentcast_client import RentCastClient
from supabase_safety import SupabaseSafetyStore


URL = 'https://abcdefghijklmnopqrst.supabase.co'
KEY = 'sb_secret_offline-test-credential'
LEDGER = '12345678-1234-4234-8234-123456789abc'


def envelope(**changes):
    return {'version': 1, 'ledger_id': LEDGER, 'allowed': True,
            'reason': 'ok', 'retry_after': None, **changes}


class Response:
    def __init__(self, body, events, status=200):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.events = events
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.events.append('admission_closed')

    def read(self, size):
        self.events.append('admission_read')
        return self.body[:size]


@pytest.fixture
def transport(monkeypatch):
    class OfflineTransport:
        def __init__(self):
            self.outcomes = []
            self.calls = []
            self.events = []
            self.handlers = []

        def build(self, *handlers):
            self.handlers.extend(handlers)
            return self

        def open(self, request, timeout):
            self.calls.append((request, timeout))
            if not self.outcomes:
                pytest.fail('Unexpected additional admission request or retry')
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return Response(outcome, self.events)

    offline = OfflineTransport()
    monkeypatch.setattr(supabase_safety, 'build_opener', offline.build)
    return offline


def configure(monkeypatch):
    monkeypatch.setenv('HOUSING_SAFETY_BACKEND', 'supabase')
    monkeypatch.setenv('HOUSING_SAFETY_SUPABASE_URL', URL)
    monkeypatch.setenv('HOUSING_SAFETY_SUPABASE_KEY', KEY)
    monkeypatch.setenv('HOUSING_SAFETY_LEDGER_ID', LEDGER)


def blocked(action, status=503):
    with pytest.raises(SafetyBlocked) as caught:
        action()
    assert caught.value.status_code == status
    assert KEY not in str(caught.value)
    return caught.value


def test_supabase_selection_never_falls_back_to_sqlite(monkeypatch, transport):
    configure(monkeypatch)
    def unexpected_sqlite(*args, **kwargs):
        pytest.fail('Supabase must never fall back to a local allowance')
    monkeypatch.setattr(housing_safety, 'SafetyStore', unexpected_sqlite)
    transport.outcomes.append(URLError('offline ' + KEY))
    blocked(housing_safety.configured_store().reserve_rentcast_request)
    assert len(transport.calls) == 1


@pytest.mark.parametrize('setting', [
    'HOUSING_SAFETY_SUPABASE_URL', 'HOUSING_SAFETY_SUPABASE_KEY',
    'HOUSING_SAFETY_LEDGER_ID',
])
def test_missing_remote_settings_fail_closed(monkeypatch, transport, setting):
    configure(monkeypatch)
    monkeypatch.delenv(setting)
    blocked(housing_safety.configured_store)
    assert transport.calls == []


@pytest.mark.parametrize('backend', ['', 'postgres', 'SUPABASE'])
def test_unknown_backend_is_not_treated_as_sqlite(monkeypatch, transport, backend):
    monkeypatch.setenv('HOUSING_SAFETY_BACKEND', backend)
    blocked(housing_safety.configured_store)
    assert transport.calls == []


@pytest.mark.parametrize('url', [
    'http://abcdefghijklmnopqrst.supabase.co',
    'https://abcdefghijklmnopqrst.supabase.co.evil.invalid',
    'https://evil.invalid',
    URL + '/rest/v1', URL + '?redirect=evil', URL + '#fragment',
    URL + ':443', URL.replace('https://', 'https://user:password@'),
    'https://short.supabase.co',
])
def test_unsafe_remote_urls_rejected_before_network(transport, url):
    blocked(lambda: SupabaseSafetyStore(url, KEY, LEDGER))
    assert transport.calls == []


@pytest.mark.parametrize('key,ledger', [
    ('', LEDGER), (None, LEDGER), (123, LEDGER), ('credential with spaces', LEDGER),
    (KEY, 'not-a-uuid'), (KEY, LEDGER.upper()), (KEY, '{' + LEDGER + '}'),
])
def test_invalid_credentials_or_noncanonical_identity_fail_closed(transport, key, ledger):
    blocked(lambda: SupabaseSafetyStore(URL, key, ledger))
    assert transport.calls == []


def test_reservation_sends_zero_allowance_and_honors_database_denial(monkeypatch, transport):
    configure(monkeypatch)
    monkeypatch.delenv('HOUSING_RENTCAST_MAX_REQUESTS_31D')
    transport.outcomes.append(envelope(allowed=False, reason='budget'))
    blocked(housing_safety.configured_store().reserve_rentcast_request, 429)
    request, timeout = transport.calls[0]
    assert request.full_url == URL + '/rest/v1/rpc/housing_safety_reserve'
    assert request.get_method() == 'POST'
    assert json.loads(request.data) == {'p_ledger_id': LEDGER, 'p_max_requests': 0}
    assert request.get_header('Apikey') == KEY
    assert request.get_header('Authorization') is None
    assert timeout == 5


def test_visitor_identity_is_hashed_before_leaving_app(transport):
    address = '198.51.100.123'
    transport.outcomes.append(envelope())
    SupabaseSafetyStore(URL, KEY, LEDGER, visitor_per_minute=2,
                        visitor_per_day=7, global_per_minute=9).check_visitor(address)
    request, _ = transport.calls[0]
    assert request.full_url.endswith('/housing_safety_admit')
    assert address.encode() not in request.data
    assert json.loads(request.data) == {
        'p_ledger_id': LEDGER,
        'p_visitor': hashlib.sha256((KEY + ':' + address).encode()).hexdigest(),
        'p_visitor_per_minute': 2, 'p_visitor_per_day': 7,
        'p_global_per_minute': 9,
    }


@pytest.mark.parametrize('reason,status', [
    ('spend', 429), ('budget', 429), ('rate', 429),
    ('settings', 503), ('storage', 503),
])
def test_database_denials_use_safe_local_errors_and_retry(transport, reason, status):
    transport.outcomes.append(envelope(allowed=False, reason=reason, retry_after=12))
    error = blocked(SupabaseSafetyStore(URL, KEY, LEDGER).reserve_rentcast_request, status)
    assert error.retry_after == 12
    if reason == 'spend':
        assert '$5' in str(error)
    assert len(transport.calls) == 1


@pytest.mark.parametrize('body', [
    b'not JSON', b'', b'x' * 65537,
    ('{"version":1,"ledger_id":"' + LEDGER + '","allowed":false,"allowed":true,'
     '"reason":"ok","retry_after":null}').encode(),
    ('{"version":2,"version":1,"ledger_id":"' + LEDGER + '","allowed":true,'
     '"reason":"ok","retry_after":null}').encode(),
    [], True, {},
    envelope(version=True), envelope(version=2),
    envelope(ledger_id='12345678-1234-4234-8234-000000000000'),
    envelope(allowed='true'), envelope(allowed=1),
    envelope(reason='spend'), envelope(allowed=False, reason='ok'),
    envelope(allowed=False, reason='unknown'), envelope(reason=[]),
    envelope(retry_after=1), envelope(allowed=False, reason='rate', retry_after=True),
    envelope(allowed=False, reason='rate', retry_after=0),
    envelope(allowed=False, reason='rate', retry_after=32 * 86400 + 1),
    {key: value for key, value in envelope().items() if key != 'retry_after'},
])
def test_ambiguous_admission_never_permits_paid_http(transport, monkeypatch, body):
    transport.outcomes.append(body)
    store = SupabaseSafetyStore(URL, KEY, LEDGER)
    import rentcast_client
    monkeypatch.setattr(rentcast_client, 'configured_store', lambda: store)
    paid_calls = []
    client = RentCastClient(api_key='offline-rentcast-key',
                           opener=lambda *args, **kwargs: paid_calls.append(args))
    blocked(lambda: client.value_estimate({'address': '123 Main St'}))
    assert paid_calls == []
    assert len(transport.calls) == 1


@pytest.mark.parametrize('failure', [
    URLError('remote error ' + KEY), TimeoutError('timeout ' + KEY),
    HTTPError(URL, 500, 'error ' + KEY, {}, io.BytesIO(KEY.encode())),
    HTTPError(URL, 302, 'redirect ' + KEY, {'Location': 'https://evil.invalid'}, None),
])
def test_transport_failure_has_no_retry_paid_call_or_secret_disclosure(transport, monkeypatch, failure):
    transport.outcomes.append(failure)
    store = SupabaseSafetyStore(URL, KEY, LEDGER)
    import rentcast_client
    monkeypatch.setattr(rentcast_client, 'configured_store', lambda: store)
    paid_calls = []
    client = RentCastClient(api_key='offline', opener=lambda *args, **kwargs: paid_calls.append(args))
    error = blocked(lambda: client.value_estimate({'address': '123 Main St'}))
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert paid_calls == []
    assert len(transport.calls) == 1


def test_remote_redirects_are_disabled_on_per_instance_opener(transport):
    SupabaseSafetyStore(URL, KEY, LEDGER)
    redirect_handlers = [handler for handler in transport.handlers
                         if isinstance(handler, HTTPRedirectHandler)]
    assert len(redirect_handlers) == 1
    assert redirect_handlers[0].redirect_request(None, None, 302, 'Found', {},
                                                'https://evil.invalid') is None


def test_successful_admission_response_finishes_before_paid_request(transport, monkeypatch):
    transport.outcomes.append(envelope())
    store = SupabaseSafetyStore(URL, KEY, LEDGER, max_requests=25)
    import rentcast_client
    monkeypatch.setattr(rentcast_client, 'configured_store', lambda: store)

    class PaidResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"price": 123456}'

    def paid_open(*args, **kwargs):
        assert transport.events == ['admission_read', 'admission_closed']
        transport.events.append('paid_request')
        return PaidResponse()

    result = RentCastClient(api_key='offline', opener=paid_open).value_estimate({'address': '123 Main St'})
    assert result['valuation']['price'] == 123456
    assert len(transport.calls) == 1
    assert transport.events[-1] == 'paid_request'


def test_status_only_calls_read_only_rpc(transport):
    expected = envelope(attempt_count32d=5, attempt_count31d=4)
    transport.outcomes.append(expected)
    assert SupabaseSafetyStore(URL, KEY, LEDGER).status() == expected
    request, _ = transport.calls[0]
    assert request.full_url.endswith('/housing_safety_status')
    assert json.loads(request.data) == {'p_ledger_id': LEDGER}


@pytest.mark.parametrize('count', [-1, True, '5', None])
def test_status_rejects_untrustworthy_counts(transport, count):
    transport.outcomes.append(envelope(attempt_count32d=count, attempt_count31d=0))
    blocked(SupabaseSafetyStore(URL, KEY, LEDGER).status)
