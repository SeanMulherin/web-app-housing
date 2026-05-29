import json
from urllib.error import HTTPError

import pytest

from rentcast_client import RentCastAuthError, RentCastClient, normalize_value_response


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode('utf-8')


def rentcast_payload():
    return {
        'price': 250000,
        'priceRangeLow': 225000,
        'priceRangeHigh': 275000,
        'subjectProperty': {
            'formattedAddress': '5500 Grand Lake Dr, San Antonio, TX 78244',
            'city': 'San Antonio',
            'state': 'TX',
            'zipCode': '78244',
            'propertyType': 'Single Family',
            'bedrooms': 3,
            'bathrooms': 2,
            'squareFootage': 1878,
            'lotSize': 8843,
        },
        'comparables': [
            {
                'formattedAddress': '5207 Pine Lake Dr, San Antonio, TX 78244',
                'status': 'Active',
                'price': 289444,
                'distance': 0.384,
                'correlation': 0.9916,
            }
        ],
    }


def test_normalize_value_response_maps_subject_and_comps():
    result = normalize_value_response(rentcast_payload())

    assert result['valuation']['price'] == 250000
    assert result['subject']['city'] == 'San Antonio'
    assert result['subject']['square_footage'] == 1878
    assert result['comparables'][0]['price'] == 289444


def test_value_estimate_requires_api_key():
    client = RentCastClient(api_key='')

    with pytest.raises(RentCastAuthError):
        client.value_estimate({'address': '5500 Grand Lake Dr, San Antonio, TX 78244'})


def test_value_estimate_calls_api_and_normalizes_response():
    def opener(request, timeout):
        assert 'avm/value' in request.full_url
        assert request.headers['X-api-key'] == 'test-key'
        return FakeResponse(rentcast_payload())

    client = RentCastClient(api_key='test-key', opener=opener)
    result = client.value_estimate({
        'address': '5500 Grand Lake Dr, San Antonio, TX 78244',
        'bedrooms': 3,
        'bathrooms': 2,
        'square_footage': 1878,
    })

    assert result['valuation']['price_range_low'] == 225000
    assert result['comparables'][0]['distance'] == 0.384


def test_value_estimate_translates_auth_errors():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 401, 'unauthorized', {}, None)

    client = RentCastClient(api_key='bad-key', opener=opener)

    with pytest.raises(RentCastAuthError):
        client.value_estimate({'address': '5500 Grand Lake Dr, San Antonio, TX 78244'})
