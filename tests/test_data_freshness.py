from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

import main
import zillow_data
from app_utils import ValidationError, normalize_analysis_request
from rentcast_client import RentCastClient, RentCastError
from valuation_service import resolve_analysis


REQUEST = {'address': '100 Main St Apt 1, Austin, TX 78701', 'period': 1, 'period_unit': 'year'}


def series_result():
    series = pd.Series([100, 110], index=pd.to_datetime(['2026-06-30', '2026-07-31']))
    return {
        'subject': {}, 'valuation': {}, 'comparables': [],
        'neighborhood': {'status': 'ok', 'listings': []}, 'warnings': [],
        'market': {'location': 'Austin, TX', 'primary_label': 'Single-family homes', 'latest_value': 110,
                   'latest_date': '2026-07-31', 'lookback_percent_change': 10, 'sfr_series': series,
                   'bedroom_series': None, 'metadata': {'sfr': {
                       'source_url': 'https://example.com/source.csv', 'cache_path': '/private/server/path',
                       'fetched_at': '2026-09-06T00:00:00+00:00', 'cache_age_seconds': 12,
                       'latest_date': '2026-07-31', 'stale': True, 'refresh_error': 'download failed',
                   }}},
    }


@pytest.fixture(autouse=True)
def clear_cache():
    main._ANALYSIS_CACHE.clear()
    yield
    main._ANALYSIS_CACHE.clear()


def test_api_freshness_force_refresh_replaces_regular_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(main, 'resolve_analysis', lambda request: calls.append(request.copy()) or series_result())
    now = [100.0]
    monkeypatch.setattr(main.time, 'monotonic', lambda: now[0])
    client = main.app.test_client()
    first = client.post('/api/analysis', json=REQUEST).json
    now[0] += 8
    cached = client.post('/api/analysis', json=REQUEST).json
    assert len(calls) == 1
    assert first['data_freshness']['cache_status'] == 'miss'
    assert cached['data_freshness']['cache_status'] == 'hit'
    assert cached['data_freshness']['cache_age_seconds'] == 8
    assert cached['data_freshness']['retrieved_at'] == first['data_freshness']['retrieved_at']
    assert cached['market']['sources']['sfr']['cache_age_seconds'] == 20
    assert 'cache_path' not in cached['market']['sources']['sfr']
    assert cached['market']['sources']['sfr']['stale'] is True
    fresh = client.post('/api/analysis', json={**REQUEST, 'force_refresh': True}).json
    assert fresh['data_freshness']['cache_status'] == 'miss'
    assert calls[-1]['force_refresh'] is True
    client.post('/api/analysis', json=REQUEST)
    assert len(calls) == 2


def test_cache_isolation_and_expiry(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(main.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(main, 'ANALYSIS_CACHE_TTL_SECONDS', 60)
    request = normalize_analysis_request(REQUEST)
    value = {'nested': {'price': 2}}
    main._cache_analysis(request, value)
    value['nested']['price'] = 9
    returned = main._get_cached_analysis(request)
    assert returned['nested']['price'] == 2
    returned['nested']['price'] = 7
    assert main._get_cached_analysis(request)['nested']['price'] == 2
    for field, value in [('property_type', 'Condo'), ('neighborhood_radius_miles', 2), ('neighborhood_max_age_days', 30)]:
        assert main._get_cached_analysis({**request, field: value}) is None
    now[0] = 70
    assert main._get_cached_analysis(request) is None


@pytest.mark.parametrize('origin', ['http://localhost:3000', 'http://127.0.0.1:3000'])
def test_preview_preflight_and_error_allow_origin(origin):
    client = main.app.test_client()
    response = client.options('/api/analysis', headers={'Origin': origin})
    assert response.status_code == 204
    assert response.headers['Access-Control-Allow-Origin'] == origin
    response = client.post('/api/analysis', json={'city': 'Austin', 'state': 'XX'}, headers={'Origin': origin})
    assert response.status_code == 400
    assert response.headers['Access-Control-Allow-Origin'] == origin


@pytest.mark.parametrize('field,value', [
    ('neighborhood_radius_miles', 0), ('neighborhood_radius_miles', 6),
    ('neighborhood_radius_miles', 'NaN'), ('square_footage', 'Infinity'),
    ('neighborhood_max_age_days', 366), ('neighborhood_max_age_days', 0),
    ('neighborhood_max_age_days', 1.5), ('force_refresh', 'yes'),
    ('force_refresh', 1), ('property_type', 'House'),
])
def test_retrieval_controls_reject_invalid_input(field, value):
    with pytest.raises(ValidationError):
        normalize_analysis_request({**REQUEST, field: value})


def test_defaults_and_numeric_zero_are_preserved():
    request = normalize_analysis_request({**REQUEST, 'bedrooms': 0, 'bathrooms': 0, 'acres': 0})
    assert request['property_type'] is None
    assert request['neighborhood_radius_miles'] == 1
    assert request['neighborhood_max_age_days'] == 180
    assert request['bedrooms'] == request['bathrooms'] == request['lot_size'] == 0


def cached_source(monkeypatch, tmp_path):
    monkeypatch.setenv('HOUSING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HOUSING_DATA_TTL_HOURS', '24')
    path = tmp_path / 'sfr.csv'
    path.write_text('RegionName,State,2026-07-31\nAustin,TX,100\n')
    return path


@pytest.mark.parametrize('failure', ['partial', 'schema', 'numeric'])
def test_source_failed_refresh_preserves_good_bytes_and_timestamp(monkeypatch, tmp_path, failure):
    path = cached_source(monkeypatch, tmp_path)
    original = path.read_bytes(), path.stat().st_mtime_ns

    def download(url, target):
        if failure == 'partial':
            target.write_text('partial bytes')
            raise OSError('connection lost')
        target.write_text('<html>Error</html>' if failure == 'schema' else 'RegionName,State,2026-07-31\nAustin,TX,bad\n')

    monkeypatch.setattr(zillow_data, 'urlretrieve', download)
    frame, metadata = zillow_data.load_source_csv('sfr', force_refresh=True)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert frame.iloc[0]['2026-07-31'] == 100
    assert metadata['stale'] is True
    assert 'refresh failed' in metadata['refresh_error']
    assert datetime.fromisoformat(metadata['fetched_at']).tzinfo is not None
    assert list(tmp_path.iterdir()) == [path]


def test_force_refresh_replaces_fresh_source_and_clears_stale_flag(monkeypatch, tmp_path):
    path = cached_source(monkeypatch, tmp_path)
    monkeypatch.setattr(zillow_data, 'urlretrieve', lambda url, target: target.write_text(
        'RegionName,State,2026-08-31\nAustin,TX,110\n'))
    frame, metadata = zillow_data.load_source_csv('sfr', force_refresh=True)
    assert frame.iloc[0]['2026-08-31'] == 110
    assert metadata['stale'] is False
    assert 'refresh_error' not in metadata
    assert list(tmp_path.iterdir()) == [path]


def test_bad_download_without_cache_raises(monkeypatch, tmp_path):
    monkeypatch.setenv('HOUSING_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(zillow_data, 'urlretrieve', lambda url, target: target.write_text('invalid'))
    with pytest.raises(zillow_data.ZillowDataError):
        zillow_data.load_source_csv('sfr')
    assert not (tmp_path / 'sfr.csv').exists()


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self):
        import json
        return json.dumps(self.payload).encode()


def raw_listing(unit, status='Active', age=1, **kwargs):
    return {'id': f'unit-{unit}', 'formattedAddress': f'100 Main St Apt {unit}, Austin, TX 78701',
            'status': status, 'price': 200000, 'lastSeenDate': (datetime.now(timezone.utc) - timedelta(days=age)).isoformat(),
            'listedDate': '2020-01-01T00:00:00Z', **kwargs}


def test_neighborhood_recency_unit_identity_and_both_status_queries():
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlsplit(request.full_url).query)
        calls.append(params)
        assert params['radius'] == ['1']
        assert params['propertyType'] == ['Condo']
        assert params['latitude'] == ['30.0'] and 'address' not in params
        assert 'daysOld' not in params
        if params['status'] == ['Active']:
            return Response([raw_listing(1), raw_listing(2), raw_listing(2), raw_listing(3, age=181),
                             raw_listing(4, lastSeenDate=None)])
        return Response([raw_listing(2, status='Inactive'), raw_listing(5, status='Inactive')])

    result = RentCastClient(api_key='test', opener=opener).neighborhood_sale_listings(
        {**REQUEST, 'neighborhood_radius_miles': 1},
        {'id': 'different-subject-id', 'formatted_address': '100 Main St #1, Austin, TX 78701',
         'property_type': 'Condo', 'latitude': 30.0, 'longitude': -97.0})
    assert len(calls) == 2
    assert [item['id'] for item in result['listings']] == ['unit-2', 'unit-5']
    assert result['excluded_older'] == result['excluded_missing_dates'] == 1
    assert result['has_more'] is False
    assert result['property_type'] == 'Condo'


def test_neighborhood_caps_at_four_requests_and_reports_truncation():
    calls = []
    def opener(request, timeout):
        params = parse_qs(urlsplit(request.full_url).query)
        calls.append(params)
        return Response([raw_listing(f'{params["status"][0]}-{params["offset"][0]}-{i}') for i in range(100)])
    result = RentCastClient(api_key='test', opener=opener).neighborhood_sale_listings(REQUEST, {})
    assert len(calls) == 4
    assert len(result['listings']) == 400
    assert result['has_more'] is True
    assert all(call['address'] == [REQUEST['address']] for call in calls)
    assert [call['offset'] for call in calls] == [['0'], ['100'], ['0'], ['100']]


def test_neighborhood_stops_after_last_seen_cutoff():
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        return Response([raw_listing(i, age=200) for i in range(100)])
    result = RentCastClient(api_key='test', opener=opener).neighborhood_sale_listings(REQUEST, {})
    assert len(calls) == 2
    assert result['has_more'] is False
    assert result['excluded_older'] == 200


def test_avm_omits_unrequested_property_type_and_missing_attributes():
    def opener(request, timeout):
        params = parse_qs(urlsplit(request.full_url).query)
        assert 'propertyType' not in params
        assert 'squareFootage' not in params and 'bedrooms' not in params
        return Response({'price': 100, 'subjectProperty': {'propertyType': 'Condo'}})
    result = RentCastClient(api_key='test', opener=opener).value_estimate(REQUEST)
    assert result['subject']['property_type'] == 'Condo'
    assert result['subject']['square_footage'] is None
    assert result['subject']['bedrooms'] is None


class Market:
    def __init__(self): self.force = []
    def get_city_series(self, location, key, force_refresh=False):
        self.force.append(force_refresh)
        return (pd.Series([100, 110], index=pd.to_datetime(['2026-06-30', '2026-07-31'])),
                {'stale': True, 'refresh_error': 'download failed', 'latest_date': '2026-07-31'})


class PartialClient:
    def value_estimate(self, request): raise RentCastError('AVM unavailable')
    def active_sale_listing(self, request):
        return {'id': 'subject-id', 'formatted_address': request['address'], 'city': 'Austin', 'state': 'TX',
                'price': 123, 'status': 'Active', 'property_type': 'Condo', 'latitude': 30, 'longitude': -97,
                'bedrooms': 0, 'square_footage': 500}
    def neighborhood_sale_listings(self, request, subject): raise RentCastError('neighborhood unavailable')


@pytest.mark.parametrize('bedrooms', [0, 3])
def test_independent_listing_survives_failed_avm_and_forces_both_sources(bedrooms):
    market = Market()
    class Client(PartialClient):
        def active_sale_listing(self, request):
            return {**super().active_sale_listing(request), 'bedrooms': bedrooms}
    result = resolve_analysis(normalize_analysis_request({**REQUEST, 'force_refresh': True}), market, Client())
    subject = result['subject']
    assert subject['listing_lookup_status'] == 'found'
    assert subject['listing_price'] == 123
    assert subject['id'] == 'subject-id' and subject['latitude'] == 30
    assert subject['property_type'] == 'Condo' and subject['bedrooms'] == bedrooms
    assert subject['attribute_sources']['square_footage'] == 'RentCast sale listing'
    assert result['valuation']['source'] == 'Zillow market benchmark'
    assert result['valuation']['retrieved_at'] is None
    assert result['neighborhood']['status'] == 'error'
    assert market.force == ([True] if bedrooms == 0 else [True, True])
    assert any('RentCast valuation:' in warning for warning in result['warnings'])
    assert any('RentCast neighborhood listings:' in warning for warning in result['warnings'])
    assert any('Zillow single-family source:' in warning for warning in result['warnings'])


@pytest.mark.parametrize('fails', [False, True])
def test_listing_missing_and_failed_are_distinct(fails):
    class Client(PartialClient):
        def active_sale_listing(self, request):
            if fails: raise RentCastError('listing timeout')
            return None
    result = resolve_analysis(normalize_analysis_request(REQUEST), Market(), Client())
    subject = result['subject']
    assert subject['listing_lookup_status'] == ('error' if fails else 'not_found')
    assert subject['listing_status'] is None
    assert subject['property_type'] is None
    assert subject['square_footage'] is None
    assert bool(subject.get('listing_lookup_error')) is fails


def test_rentcast_retrieval_timestamp_preserved_through_service():
    from rentcast_client import normalize_value_response
    observed = normalize_value_response({'price': 100, 'subjectProperty': {'city': 'Austin', 'state': 'TX'}})
    timestamp = datetime.fromisoformat(observed['valuation']['retrieved_at'])
    assert timestamp.tzinfo == timezone.utc
    assert observed['valuation']['as_of'] == timestamp.date().isoformat()
    class Client:
        def value_estimate(self, request): return observed
    result = resolve_analysis(normalize_analysis_request(REQUEST), Market(), Client())
    assert result['valuation']['retrieved_at'] == timestamp.isoformat()
