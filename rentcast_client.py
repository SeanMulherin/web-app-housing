import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app_utils import ValidationError


RENTCAST_VALUE_URL = 'https://api.rentcast.io/v1/avm/value'


class RentCastError(RuntimeError):
    pass


class RentCastAuthError(RentCastError):
    pass


def _timeout_seconds():
    raw_timeout = os.getenv('RENTCAST_TIMEOUT_SECONDS', '10')
    try:
        return max(1, float(raw_timeout))
    except ValueError:
        return 10


class RentCastClient:
    def __init__(self, api_key=None, opener=None):
        self.api_key = api_key if api_key is not None else os.getenv('RENTCAST_API_KEY')
        self.opener = opener or urlopen

    def value_estimate(self, analysis_request):
        if not self.api_key:
            raise RentCastAuthError('Set RENTCAST_API_KEY to enable address-level valuation and comparables.')
        if not analysis_request.get('address'):
            raise ValidationError('Address is required for RentCast value estimates in this version.')

        params = {
            'address': analysis_request['address'],
            'propertyType': 'Single Family',
            'compCount': 15,
            'lookupSubjectAttributes': 'true',
        }
        optional_params = {
            'bedrooms': analysis_request.get('bedrooms'),
            'bathrooms': analysis_request.get('bathrooms'),
            'squareFootage': analysis_request.get('square_footage'),
            'maxRadius': analysis_request.get('max_radius'),
            'daysOld': analysis_request.get('days_old'),
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})

        payload = self._get_json(RENTCAST_VALUE_URL, params)
        return normalize_value_response(payload)

    def _get_json(self, url, params):
        request_url = f'{url}?{urlencode(params)}'
        request = Request(
            request_url,
            headers={'accept': 'application/json', 'X-Api-Key': self.api_key},
            method='GET',
        )
        try:
            with self.opener(request, timeout=_timeout_seconds()) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise RentCastAuthError('RentCast rejected the API key.') from exc
            raise RentCastError(f'RentCast request failed with status {exc.code}.') from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RentCastError('RentCast value data could not be loaded right now.') from exc


def normalize_value_response(payload):
    subject = normalize_property(payload.get('subjectProperty') or {})
    comparables = [normalize_property(comp) for comp in payload.get('comparables') or []]
    return {
        'valuation': {
            'price': payload.get('price'),
            'price_range_low': payload.get('priceRangeLow'),
            'price_range_high': payload.get('priceRangeHigh'),
            'source': 'RentCast AVM',
            'as_of': datetime.now(timezone.utc).date().isoformat(),
        },
        'subject': subject,
        'comparables': comparables,
    }


def normalize_property(raw_property):
    return {
        'id': raw_property.get('id'),
        'formatted_address': raw_property.get('formattedAddress'),
        'address_line_1': raw_property.get('addressLine1'),
        'city': raw_property.get('city'),
        'state': raw_property.get('state'),
        'zip_code': raw_property.get('zipCode'),
        'county': raw_property.get('county'),
        'latitude': raw_property.get('latitude'),
        'longitude': raw_property.get('longitude'),
        'property_type': raw_property.get('propertyType'),
        'bedrooms': raw_property.get('bedrooms'),
        'bathrooms': raw_property.get('bathrooms'),
        'square_footage': raw_property.get('squareFootage'),
        'lot_size': raw_property.get('lotSize'),
        'year_built': raw_property.get('yearBuilt'),
        'last_sale_date': raw_property.get('lastSaleDate'),
        'last_sale_price': raw_property.get('lastSalePrice'),
        'status': raw_property.get('status'),
        'price': raw_property.get('price'),
        'listed_date': raw_property.get('listedDate'),
        'last_seen_date': raw_property.get('lastSeenDate'),
        'days_on_market': raw_property.get('daysOnMarket'),
        'distance': raw_property.get('distance'),
        'days_old': raw_property.get('daysOld'),
        'correlation': raw_property.get('correlation'),
    }
