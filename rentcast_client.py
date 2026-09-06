import json
import math
import re
import os
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app_utils import ValidationError


RENTCAST_VALUE_URL = 'https://api.rentcast.io/v1/avm/value'
RENTCAST_SALE_LISTINGS_URL = 'https://api.rentcast.io/v1/listings/sale'


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
            'compCount': 15,
            'lookupSubjectAttributes': 'true',
        }
        optional_params = {
            'propertyType': analysis_request.get('property_type'),
            'bedrooms': analysis_request.get('bedrooms'),
            'bathrooms': analysis_request.get('bathrooms'),
            'squareFootage': analysis_request.get('square_footage'),
            'maxRadius': analysis_request.get('max_radius'),
            'daysOld': analysis_request.get('days_old'),
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})

        payload = self._get_json(RENTCAST_VALUE_URL, params)
        return normalize_value_response(payload)

    def active_sale_listing(self, analysis_request):
        if not self.api_key:
            raise RentCastAuthError('Set RENTCAST_API_KEY to enable address-level valuation and comparables.')
        if not analysis_request.get('address'):
            raise ValidationError('Address is required for RentCast sale listing searches.')

        payload = self._get_json(RENTCAST_SALE_LISTINGS_URL, {'address': analysis_request['address'], 'status': 'Active', 'limit': 100})
        if not isinstance(payload, list):
            raise RentCastError('RentCast returned an invalid sale-listing response.')
        listings = [normalize_property(listing) for listing in payload]
        return next(
            (listing for listing in listings if str(listing.get('status')).lower() == 'active' and listing.get('price')),
            None,
        )

    def neighborhood_sale_listings(self, analysis_request, subject):
        if not self.api_key:
            raise RentCastAuthError('Set RENTCAST_API_KEY to enable neighborhood listings.')
        now = datetime.now(timezone.utc)
        radius = analysis_request.get('neighborhood_radius_miles', 1.0)
        max_age = analysis_request.get('neighborhood_max_age_days', 180)
        property_type = subject.get('property_type')
        params = {'radius': radius, 'limit': 100}
        if subject.get('latitude') is not None and subject.get('longitude') is not None:
            params.update(latitude=subject['latitude'], longitude=subject['longitude'])
        elif analysis_request.get('address'):
            params['address'] = analysis_request['address']
        else:
            raise RentCastError('An address or resolved coordinates are required for neighborhood listings.')
        if property_type:
            params['propertyType'] = property_type
        result = {
            'listings': [], 'source': 'RentCast sale listings', 'status': 'ok',
            'radius_miles': radius, 'max_age_days': max_age, 'property_type': property_type,
            'retrieved_at': now.isoformat(), 'has_more': False, 'excluded_older': 0,
            'excluded_missing_dates': 0,
        }
        cutoff = now - timedelta(days=max_age)
        subject_keys = property_identity_keys(subject)
        subject_keys.update(property_identity_keys({'formatted_address': analysis_request.get('address')}))
        seen = set()
        # The documented API sorts lastSeenDate descending. Two pages per status
        # bound cost, while a cutoff crossing ends that status search early.
        # daysOld on /listings means days since LISTED, so recency is checked below.
        for status in ('Active', 'Inactive'):
            for page in range(2):
                payload = self._get_json(RENTCAST_SALE_LISTINGS_URL, {
                    **params, 'status': status, 'offset': page * 100,
                })
                if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
                    raise RentCastError('RentCast returned an invalid neighborhood-listing response.')
                crossed_cutoff = False
                for raw in payload:
                    listing = normalize_property(raw)
                    last_seen = parse_source_date(listing.get('last_seen_date'))
                    if last_seen is None:
                        result['excluded_missing_dates'] += 1
                        continue
                    if last_seen < cutoff:
                        result['excluded_older'] += 1
                        crossed_cutoff = True
                        continue
                    keys = property_identity_keys(listing)
                    if not keys or keys.intersection(subject_keys) or keys.intersection(seen):
                        continue
                    seen.update(keys)
                    if listing.get('distance') is None:
                        listing['distance'] = distance_miles(subject, listing)
                    result['listings'].append(listing)
                if len(payload) < 100 or crossed_cutoff:
                    break
                if page == 1:
                    # No count/header request: a full final page may have more matches.
                    result['has_more'] = True
        return result

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
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RentCastError('RentCast data could not be loaded right now.') from exc


def normalize_value_response(payload):
    if not isinstance(payload, dict):
        raise RentCastError('RentCast returned an invalid valuation response.')
    retrieved_at = datetime.now(timezone.utc)
    subject = normalize_property(payload.get('subjectProperty') or {})
    comparables = [normalize_property(comp) for comp in payload.get('comparables') or []]
    return {
        'valuation': {
            'price': payload.get('price'),
            'price_range_low': payload.get('priceRangeLow'),
            'price_range_high': payload.get('priceRangeHigh'),
            'source': 'RentCast AVM',
            'as_of': retrieved_at.date().isoformat(),
            'retrieved_at': retrieved_at.isoformat(),
        },
        'subject': subject,
        'comparables': comparables,
    }


def normalize_property(raw_property):
    return {
        'id': raw_property.get('id'),
        'formatted_address': raw_property.get('formattedAddress'),
        'address_line_1': raw_property.get('addressLine1'),
        'address_line_2': raw_property.get('addressLine2'),
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


def parse_source_date(value):
    try:
        timestamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def property_identity_keys(property_record):
    keys = set()
    if property_record.get('id'):
        keys.add('id:' + str(property_record['id']).lower())
    address = property_record.get('formatted_address')
    if not address and property_record.get('address_line_1'):
        address = ', '.join(str(property_record.get(field) or '') for field in (
            'address_line_1', 'address_line_2', 'city', 'state', 'zip_code'
        ))
    if address:
        # Preserve the full address INCLUDING unit number; never collapse a building.
        address = re.sub(r'\b(?:apt|apartment|unit|suite)\s*|#\s*', ' unit ', str(address).lower())
        address = re.sub(r'[^a-z0-9]', '', address)
        keys.add('address:' + address)
    return keys


def distance_miles(subject, listing):
    try:
        lat1, lon1, lat2, lon2 = (float(value) for value in (
            subject.get('latitude'), subject.get('longitude'), listing.get('latitude'), listing.get('longitude')
        ))
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        return round(3958.7613 * 2 * math.asin(min(1, math.sqrt(value))), 4)
    except (TypeError, ValueError):
        return None
