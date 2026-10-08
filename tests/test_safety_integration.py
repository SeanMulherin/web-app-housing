import sqlite3
from urllib.error import HTTPError

import pytest

import main
from app_utils import normalize_analysis_request
from housing_safety import SafetyBlocked, configured_store
from rentcast_client import RentCastClient, RentCastError, _RejectRedirects
from test_flask_routes import sample_analysis
from test_rentcast_client import FakeResponse, rentcast_payload
from valuation_service import resolve_analysis


REQUEST = {'address': '123 Main St, Austin, TX 78701', 'period': '10', 'period_unit': 'year'}


def attempts(path):
    with sqlite3.connect(path) as connection:
        return connection.execute('SELECT COUNT(*) FROM rentcast_attempts').fetchone()[0]


@pytest.mark.parametrize('method', ['value_estimate', 'active_sale_listing', 'neighborhood_sale_listings'])
def test_every_paid_method_blocks_before_network(monkeypatch, method, isolated_safety_ledger):
    monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '0')
    calls = []
    client = RentCastClient(api_key='fake', opener=lambda *args, **kwargs: calls.append(args))
    arguments = (REQUEST, {}) if method == 'neighborhood_sale_listings' else (REQUEST,)
    with pytest.raises(SafetyBlocked) as denial:
        getattr(client, method)(*arguments)
    assert denial.value.status_code == 429
    assert calls == []
    assert attempts(isolated_safety_ledger) == 0


def test_budget_commits_before_network_and_failed_attempts_are_not_refunded(monkeypatch, isolated_safety_ledger):
    monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '1')
    def fail(request, timeout):
        assert attempts(isolated_safety_ledger) == 1
        raise HTTPError(request.full_url, 500, 'failed', {}, None)
    client = RentCastClient(api_key='fake', opener=fail)
    with pytest.raises(RentCastError):
        client.value_estimate(REQUEST)
    with pytest.raises(SafetyBlocked):
        RentCastClient(api_key='fake', opener=fail).active_sale_listing(REQUEST)
    assert attempts(isolated_safety_ledger) == 1


@pytest.mark.parametrize('route', ['/api/analysis', '/forecast'])
@pytest.mark.parametrize('failure', ['budget', 'storage'])
def test_both_routes_propagate_safety_errors_without_caching(monkeypatch, route, failure, isolated_safety_ledger):
    if failure == 'budget':
        monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '0')
        status = 429
    else:
        monkeypatch.setenv('HOUSING_SAFETY_DB_PATH', str(isolated_safety_ledger.parent / 'missing.sqlite3'))
        status = 503
    calls = []
    def analyze(request):
        return RentCastClient(api_key='fake', opener=lambda *args, **kwargs: calls.append(args)).value_estimate(request)
    monkeypatch.setattr(main, 'resolve_analysis', analyze)
    options = {'json': REQUEST} if route == '/api/analysis' else {'data': REQUEST}
    response = main.app.test_client().post(route, **options)
    assert response.status_code == status
    assert response.headers['Cache-Control'] == 'no-store'
    assert calls == []
    assert main._ANALYSIS_CACHE == {}


@pytest.mark.parametrize('route', ['/api/analysis', '/forecast'])
def test_public_refresh_cannot_reach_paid_calls(monkeypatch, route):
    calls = []
    monkeypatch.setattr(main, 'resolve_analysis', lambda request: calls.append(request))
    options = {'json': {**REQUEST, 'force_refresh': True}} if route == '/api/analysis' else {'data': {**REQUEST, 'force_refresh': 'true'}}
    response = main.app.test_client().post(route, **options)
    assert response.status_code == 403
    assert calls == []


def test_visitor_limit_shared_across_routes_and_forwarded_headers_cannot_evade_it(monkeypatch):
    monkeypatch.setenv('HOUSING_VISITOR_REQUESTS_PER_MINUTE', '1')
    monkeypatch.setattr(main, 'resolve_analysis', lambda request: sample_analysis())
    client = main.app.test_client()
    assert client.post('/api/analysis', json=REQUEST, headers={'X-Forwarded-For': '203.0.113.1'}).status_code == 200
    response = client.post('/forecast', data=REQUEST, headers={'X-Forwarded-For': '203.0.113.2'})
    assert response.status_code == 429
    assert int(response.headers['Retry-After']) > 0


def test_cached_analysis_uses_no_paid_allowance(monkeypatch, isolated_safety_ledger):
    monkeypatch.setattr(main, 'resolve_analysis', lambda request: sample_analysis())
    client = main.app.test_client()
    assert client.post('/api/analysis', json=REQUEST).headers['X-Analysis-Cache'] == 'MISS'
    monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '0')
    def unexpected(request):
        pytest.fail('Cache hit must not resolve a new analysis')
    monkeypatch.setattr(main, 'resolve_analysis', unexpected)
    response = client.post('/api/analysis', json=REQUEST)
    assert response.status_code == 200
    assert response.headers['X-Analysis-Cache'] == 'HIT'
    assert attempts(isolated_safety_ledger) == 0


@pytest.mark.parametrize('blocked_method', ['value_estimate', 'active_sale_listing', 'neighborhood_sale_listings'])
def test_service_does_not_swallow_denial_from_parallel_lookup_or_neighborhood(blocked_method):
    class FakeClient:
        def value_estimate(self, request):
            if blocked_method == 'value_estimate':
                raise SafetyBlocked('blocked', 429)
            return {'subject': {'city': 'Austin', 'state': 'TX'}}
        def active_sale_listing(self, request):
            if blocked_method == 'active_sale_listing':
                raise SafetyBlocked('blocked', 503)
            return None
        def neighborhood_sale_listings(self, request, subject):
            raise SafetyBlocked('blocked', 429)
    with pytest.raises(SafetyBlocked):
        resolve_analysis(normalize_analysis_request(REQUEST), rentcast_client=FakeClient())


def test_default_zero_budget_and_missing_configuration_fail_closed(monkeypatch):
    monkeypatch.delenv('HOUSING_RENTCAST_MAX_REQUESTS_31D')
    with pytest.raises(SafetyBlocked) as denial:
        configured_store().reserve_rentcast_request()
    assert denial.value.status_code == 429
    monkeypatch.delenv('HOUSING_SAFETY_DB_PATH')
    with pytest.raises(SafetyBlocked) as denial:
        configured_store()
    assert denial.value.status_code == 503


def test_invalid_configuration_does_not_disable_safety(monkeypatch):
    monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '-1')
    with pytest.raises(SafetyBlocked) as denial:
        configured_store()
    assert denial.value.status_code == 503


def test_redirects_are_not_followed():
    assert _RejectRedirects().redirect_request(None, None, 302, 'Found', {}, 'https://example.invalid') is None


def test_preflight_does_not_need_a_ledger_or_paid_calls(monkeypatch):
    monkeypatch.delenv('HOUSING_SAFETY_DB_PATH')
    response = main.app.test_client().options('/api/analysis', headers={'Origin': 'https://seanmulherin.github.io'})
    assert response.status_code == 204
    assert response.headers['X-Housing-Safety-Version'] == '2026-10-08-v1'
    assert response.headers['Access-Control-Allow-Origin'] == 'https://seanmulherin.github.io'


def test_finance_remains_available_without_housing_storage(monkeypatch):
    import finance_market_api
    monkeypatch.delenv('HOUSING_SAFETY_DB_PATH')
    expected = {'rate': 0.04, 'source': 'offline test'}
    monkeypatch.setattr(finance_market_api, '_risk_free_payload', lambda: expected)
    response = main.app.test_client().get('/api/finance/market?kind=riskfree')
    assert response.status_code == 200
    assert response.json == expected
    assert 'X-Housing-Safety-Version' not in response.headers


@pytest.mark.parametrize('route', ['/api/analysis', '/forecast'])
def test_oversize_input_rejected_before_analysis(monkeypatch, route):
    calls = []
    monkeypatch.setattr(main, 'resolve_analysis', lambda request: calls.append(request))
    response = main.app.test_client().post(route, data='x' * 8193, content_type='application/json' if route == '/api/analysis' else 'application/x-www-form-urlencoded')
    assert response.status_code == 413
    assert calls == []
