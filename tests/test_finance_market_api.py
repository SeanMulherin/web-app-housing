import json

import finance_market_api as market_module
from main import app


def setup_function():
    market_module._CACHE.clear()


def test_market_route_serializes_adjusted_history(monkeypatch):
    timestamps = list(range(1_700_000_000, 1_700_000_000 + 12 * 86400, 86400))
    payload = {
        'chart': {
            'result': [{
                'timestamp': timestamps,
                'indicators': {'adjclose': [{'adjclose': [100 + index for index in range(12)]}]},
            }],
        },
    }
    monkeypatch.setattr(market_module, '_fetch_text', lambda url, accept: json.dumps(payload))

    response = app.test_client().get(
        '/api/finance/market?symbol=voo&range=1Y',
        headers={'Origin': 'https://seanmulherin.github.io'},
    )

    assert response.status_code == 200
    assert response.json['symbol'] == 'VOO'
    assert len(response.json['rows']) == 12
    assert response.headers['Access-Control-Allow-Origin'] == 'https://seanmulherin.github.io'


def test_market_route_rejects_invalid_inputs():
    client = app.test_client()
    assert client.get('/api/finance/market?symbol=bad symbol&range=1Y').status_code == 400
    assert client.get('/api/finance/market?symbol=VOO&range=20Y').status_code == 400


def test_risk_free_route_uses_latest_fred_observation(monkeypatch):
    fred_csv = 'DATE,DGS10\n2026-09-07,\n2026-09-08,4.80\n'
    monkeypatch.setattr(market_module, '_fetch_text', lambda url, accept: fred_csv)

    response = app.test_client().get('/api/finance/market?kind=riskfree')

    assert response.status_code == 200
    assert response.json == {
        'date': '2026-09-08',
        'isFallback': False,
        'percent': 4.8,
        'rate': 0.048,
        'source': 'FRED DGS10',
    }


def test_risk_free_route_falls_back_when_fred_fails(monkeypatch):
    def fail(url, accept):
        raise RuntimeError('upstream unavailable')

    monkeypatch.setattr(market_module, '_fetch_text', fail)
    response = app.test_client().get('/api/finance/market?kind=riskfree')

    assert response.status_code == 200
    assert response.json['rate'] == 0.04
    assert response.json['isFallback'] is True
