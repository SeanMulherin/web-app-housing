import csv
import io
import json
import re
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from flask import Blueprint, jsonify, request


finance_market_api = Blueprint('finance_market_api', __name__)

RANGE_DAYS = {
    '1M': 32,
    '3M': 95,
    '1Y': 370,
    '3Y': 1100,
    '5Y': 1830,
}
DEFAULT_RISK_FREE_RATE = 0.04
_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.^=-]{1,12}$')
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _fetch_text(url, accept):
    upstream_request = Request(
        url,
        headers={
            'Accept': accept,
            'User-Agent': 'Mozilla/5.0 Signal-Allocation-Lab/1.0',
        },
    )
    with urlopen(upstream_request, timeout=12) as response:
        return response.read().decode('utf-8')


def _cache_get(key):
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if not cached:
            return None
        expires_at, payload = cached
        if expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return payload


def _cache_put(key, payload, ttl_seconds):
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic() + ttl_seconds, payload)


def _risk_free_payload():
    key = ('riskfree',)
    cached = _cache_get(key)
    if cached:
        return cached

    try:
        text = _fetch_text(
            'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10',
            'text/csv',
        )
        rows = list(csv.reader(io.StringIO(text)))
        latest = next(
            (
                (date, float(value))
                for date, value, *_ in reversed(rows[1:])
                if value and value != '.'
            ),
            None,
        )
        if latest is None:
            raise ValueError('FRED did not return a current observation.')
        date, percent = latest
        payload = {
            'rate': percent / 100,
            'percent': percent,
            'date': date,
            'source': 'FRED DGS10',
            'isFallback': False,
        }
        _cache_put(key, payload, 3600)
        return payload
    except Exception:
        return {
            'rate': DEFAULT_RISK_FREE_RATE,
            'percent': DEFAULT_RISK_FREE_RATE * 100,
            'date': None,
            'source': '4% fallback',
            'isFallback': True,
        }


def _market_payload(symbol, selected_range):
    key = ('market', symbol, selected_range)
    cached = _cache_get(key)
    if cached:
        return cached

    period2 = int(time.time())
    period1 = period2 - RANGE_DAYS[selected_range] * 86400
    endpoint = (
        'https://query1.finance.yahoo.com/v8/finance/chart/'
        f'{quote(symbol, safe="")}?period1={period1}&period2={period2}'
        '&interval=1d&events=history&includeAdjustedClose=true'
    )
    upstream = json.loads(_fetch_text(endpoint, 'application/json'))
    results = (upstream.get('chart') or {}).get('result') or []
    if not results:
        raise ValueError('No adjusted price history was returned.')

    result = results[0]
    timestamps = result.get('timestamp') or []
    indicators = result.get('indicators') or {}
    adjusted = ((indicators.get('adjclose') or [{}])[0].get('adjclose') or [])
    closes = adjusted or ((indicators.get('quote') or [{}])[0].get('close') or [])
    rows = []
    for timestamp, close in zip(timestamps, closes):
        if not isinstance(close, (int, float)) or close <= 0:
            continue
        rows.append({
            'date': time.strftime('%Y-%m-%d', time.gmtime(timestamp)),
            'close': close,
        })
    if len(rows) < 10:
        raise ValueError('Not enough market history was returned.')

    payload = {'symbol': symbol, 'range': selected_range, 'rows': rows}
    _cache_put(key, payload, 900)
    return payload


@finance_market_api.route('/api/finance/market', methods=['GET', 'OPTIONS'])
def market_data():
    if request.method == 'OPTIONS':
        return '', 204

    if request.args.get('kind') == 'riskfree':
        return jsonify(_risk_free_payload())

    symbol = request.args.get('symbol', '').strip().upper()
    selected_range = request.args.get('range', '1Y')
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        return jsonify({'error': 'Invalid ticker symbol.'}), 400
    if selected_range not in RANGE_DAYS:
        return jsonify({'error': 'Invalid date range.'}), 400

    try:
        return jsonify(_market_payload(symbol, selected_range))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception:
        return jsonify({'error': 'The market feed is temporarily unavailable.'}), 502
