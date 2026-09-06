import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app_utils import ValidationError, normalize_location, percent_change
from rentcast_client import RentCastAuthError, RentCastClient, RentCastError
from zillow_data import ZillowDataError, ZillowMarketData, bedroom_series_key


def resolve_analysis(analysis_request, market_data=None, rentcast_client=None):
    market_data = market_data or ZillowMarketData()
    rentcast_client = rentcast_client or RentCastClient()
    warnings = []

    rentcast_result = _get_rentcast_result(analysis_request, rentcast_client, warnings)
    subject = _subject_from_request_and_rentcast(analysis_request, rentcast_result)
    location = _resolve_location(analysis_request, subject)

    neighborhood = _get_neighborhood_result(analysis_request, subject, rentcast_client, warnings)
    market = _get_market_result(location, analysis_request, subject, market_data, warnings)
    valuation = _valuation_from_rentcast_and_market(rentcast_result, market)
    comparables = rentcast_result.get('comparables', []) if rentcast_result else []

    return {
        'subject': subject,
        'valuation': valuation,
        'market': market,
        'comparables': comparables,
        'neighborhood': neighborhood,
        'warnings': warnings,
    }


def _get_rentcast_result(analysis_request, rentcast_client, warnings):
    if not analysis_request.get('address'):
        warnings.append('No address was supplied, so the app is showing a market benchmark instead of an address-level AVM.')
        return {'listing_lookup_status': 'unavailable'}

    listing_lookup = getattr(rentcast_client, 'active_sale_listing', None)
    result = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        valuation_future = executor.submit(rentcast_client.value_estimate, analysis_request)
        listing_future = executor.submit(listing_lookup, analysis_request) if listing_lookup else None
        try:
            result = valuation_future.result() or {}
        except (RentCastAuthError, RentCastError) as exc:
            warnings.append(f'RentCast valuation: {exc}')
            result['valuation_lookup_error'] = str(exc)
        result['listing_lookup_status'] = 'unavailable'
        if listing_future:
            try:
                listing = listing_future.result()
                result['active_listing'] = listing
                result['listing_lookup_status'] = 'found' if listing else 'not_found'
            except (RentCastAuthError, RentCastError) as exc:
                warnings.append(f'RentCast subject listing: {exc}')
                result['listing_lookup_status'] = 'error'
                result['listing_lookup_error'] = str(exc)
    return result


def _subject_from_request_and_rentcast(analysis_request, rentcast_result):
    result = rentcast_result or {}
    avm_subject = result.get('subject') or {}
    active_listing = result.get('active_listing') or {}
    subject = {}
    sources = {}
    for key in ('id', 'formatted_address', 'address_line_1', 'address_line_2', 'city', 'state', 'zip_code',
                'county', 'latitude', 'longitude', 'property_type', 'bedrooms', 'bathrooms',
                'square_footage', 'lot_size', 'year_built', 'last_sale_date', 'last_sale_price'):
        request_key = 'address' if key == 'formatted_address' else key
        if active_listing.get(key) is not None:
            subject[key] = active_listing[key]
            sources[key] = 'RentCast sale listing'
        elif avm_subject.get(key) is not None:
            subject[key] = avm_subject[key]
            supplied_to_avm = key in {'property_type', 'bedrooms', 'bathrooms', 'square_footage'} and analysis_request.get(key) is not None
            sources[key] = 'User supplied (used by RentCast AVM)' if supplied_to_avm else 'RentCast AVM subject'
        else:
            subject[key] = analysis_request.get(request_key)
            if subject[key] is not None:
                sources[key] = 'User supplied'
    subject.update({
        'listing_status': active_listing.get('status'),
        'listing_price': active_listing.get('price'),
        'listed_date': active_listing.get('listed_date'),
        'listing_last_seen_date': active_listing.get('last_seen_date'),
        'days_on_market': active_listing.get('days_on_market'),
        'listing_lookup_status': result.get('listing_lookup_status', 'unavailable'),
        'attribute_sources': sources,
    })
    if result.get('listing_lookup_error'):
        subject['listing_lookup_error'] = result['listing_lookup_error']
    return subject


def _get_neighborhood_result(analysis_request, subject, rentcast_client, warnings):
    neighborhood = {
        'listings': [], 'source': 'RentCast sale listings', 'status': 'unavailable',
        'radius_miles': analysis_request.get('neighborhood_radius_miles', 1.0),
        'max_age_days': analysis_request.get('neighborhood_max_age_days', 180),
        'property_type': subject.get('property_type'), 'retrieved_at': None,
        'has_more': False, 'excluded_older': 0, 'excluded_missing_dates': 0,
    }
    lookup = getattr(rentcast_client, 'neighborhood_sale_listings', None)
    if not lookup or not analysis_request.get('address'):
        return neighborhood
    try:
        neighborhood = lookup(analysis_request, subject)
        if neighborhood.get('has_more'):
            warnings.append('Neighborhood listings reached the bounded search limit; this is a truncated sample, not every listing in the area.')
        if neighborhood.get('excluded_missing_dates'):
            warnings.append('Neighborhood listings without a usable last-seen date were excluded because their freshness could not be checked.')
        return neighborhood
    except (RentCastAuthError, RentCastError) as exc:
        neighborhood.update(status='error', error=str(exc), retrieved_at=datetime.now(timezone.utc).isoformat())
        warnings.append(f'RentCast neighborhood listings: {exc}')
        return neighborhood


def _resolve_location(analysis_request, subject):
    if analysis_request.get('location'):
        return analysis_request['location']
    if subject.get('city') and subject.get('state'):
        return normalize_location(subject['city'], subject['state'])
    raise ValidationError('City and state are required when address lookup cannot resolve a market.')


def _get_market_result(location, analysis_request, subject, market_data, warnings):
    source_options = {'force_refresh': True} if analysis_request.get('force_refresh') else {}
    try:
        sfr_series, sfr_metadata = market_data.get_city_series(location, 'sfr', **source_options)
    except (ValidationError, ZillowDataError):
        raise

    bedroom_series = None
    bedroom_metadata = None
    key = bedroom_series_key(subject.get('bedrooms'))
    if key:
        try:
            bedroom_series, bedroom_metadata = market_data.get_city_series(location, key, **source_options)
            warnings.append('Bedroom-specific Zillow history includes all Zillow home types, so use it as a bedroom benchmark rather than a pure single-family segment.')
        except (ValidationError, ZillowDataError):
            warnings.append('Bedroom-specific Zillow history was not available, so the comparison uses all single-family homes.')

    for source_name, metadata in (('single-family', sfr_metadata), ('bedroom', bedroom_metadata)):
        if metadata and metadata.get('stale'):
            warnings.append(f'Zillow {source_name} source: {metadata.get("refresh_error", "using a stale cached source")}')

    primary_series = bedroom_series if bedroom_series is not None else sfr_series
    primary_label = _series_label(key) if bedroom_series is not None else 'Single-family homes'
    lookback_months = analysis_request.get('period_months', 120)

    return {
        'location': location,
        'sfr_series': sfr_series,
        'bedroom_series': bedroom_series,
        'primary_series': primary_series,
        'primary_label': primary_label,
        'latest_value': _last_value(primary_series),
        'latest_date': primary_series.index[-1].date().isoformat(),
        'lookback_percent_change': _lookback_change(primary_series, lookback_months),
        'sfr_latest_value': _last_value(sfr_series),
        'bedroom_latest_value': _last_value(bedroom_series) if bedroom_series is not None else None,
        'metadata': {
            'sfr': sfr_metadata,
            'bedroom': bedroom_metadata,
        },
    }


def _series_label(series_key):
    if series_key == 'bedroom_5':
        return '5+ bedroom homes'
    if series_key and series_key.startswith('bedroom_'):
        return f'{series_key.split("_")[1]} bedroom homes'
    return 'Single-family homes'


def _last_value(series):
    return float(series.dropna().iloc[-1])


def _lookback_change(series, months):
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return None
    reference_index = max(0, len(clean_series) - months - 1)
    reference = float(clean_series.iloc[reference_index])
    if reference == 0 or math.isnan(reference):
        return None
    return percent_change(float(clean_series.iloc[-1]), reference)


def _valuation_from_rentcast_and_market(rentcast_result, market):
    valuation = rentcast_result.get('valuation', {}) if rentcast_result else {}
    price = valuation.get('price')
    market_value = market.get('latest_value')
    if price is None:
        return {
            'price': market_value,
            'price_range_low': None,
            'price_range_high': None,
            'source': 'Zillow market benchmark',
            'as_of': market.get('latest_date'),
            'retrieved_at': None,
            'relative_to_market_percent': None,
            'market_value': market_value,
        }

    return {
        'price': price,
        'price_range_low': valuation.get('price_range_low'),
        'price_range_high': valuation.get('price_range_high'),
        'source': valuation.get('source'),
        'as_of': valuation.get('as_of') or market.get('latest_date'),
        'retrieved_at': valuation.get('retrieved_at'),
        'relative_to_market_percent': percent_change(price, market_value) if price and market_value else None,
        'market_value': market_value,
    }
