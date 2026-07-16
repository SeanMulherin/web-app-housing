import math

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

    market = _get_market_result(location, analysis_request, subject, market_data, warnings)
    valuation = _valuation_from_rentcast_and_market(rentcast_result, market)
    comparables = rentcast_result.get('comparables', []) if rentcast_result else []

    return {
        'subject': subject,
        'valuation': valuation,
        'market': market,
        'comparables': comparables,
        'warnings': warnings,
    }


def _get_rentcast_result(analysis_request, rentcast_client, warnings):
    if not analysis_request.get('address'):
        warnings.append('No address was supplied, so the app is showing a market benchmark instead of an address-level AVM.')
        return None

    try:
        return rentcast_client.value_estimate(analysis_request)
    except RentCastAuthError as exc:
        warnings.append(str(exc))
        return None
    except RentCastError as exc:
        warnings.append(str(exc))
        return None


def _subject_from_request_and_rentcast(analysis_request, rentcast_result):
    subject = rentcast_result.get('subject', {}) if rentcast_result else {}
    return {
        'formatted_address': subject.get('formatted_address') or analysis_request.get('address'),
        'city': subject.get('city') or analysis_request.get('city'),
        'state': subject.get('state') or analysis_request.get('state'),
        'zip_code': subject.get('zip_code'),
        'county': subject.get('county'),
        'property_type': subject.get('property_type') or 'Single Family',
        'bedrooms': subject.get('bedrooms') if subject.get('bedrooms') is not None else analysis_request.get('bedrooms'),
        'bathrooms': subject.get('bathrooms') if subject.get('bathrooms') is not None else analysis_request.get('bathrooms'),
        'square_footage': subject.get('square_footage') if subject.get('square_footage') is not None else analysis_request.get('square_footage'),
        'lot_size': subject.get('lot_size') if subject.get('lot_size') is not None else analysis_request.get('lot_size'),
        'year_built': subject.get('year_built') if subject.get('year_built') is not None else analysis_request.get('year_built'),
        'last_sale_date': subject.get('last_sale_date'),
        'last_sale_price': subject.get('last_sale_price'),
    }


def _resolve_location(analysis_request, subject):
    if analysis_request.get('location'):
        return analysis_request['location']
    if subject.get('city') and subject.get('state'):
        return normalize_location(subject['city'], subject['state'])
    raise ValidationError('City and state are required when address lookup cannot resolve a market.')


def _get_market_result(location, analysis_request, subject, market_data, warnings):
    try:
        sfr_series, sfr_metadata = market_data.get_city_series(location, 'sfr')
    except (ValidationError, ZillowDataError):
        raise

    bedroom_series = None
    bedroom_metadata = None
    key = bedroom_series_key(subject.get('bedrooms'))
    if key:
        try:
            bedroom_series, bedroom_metadata = market_data.get_city_series(location, key)
            warnings.append('Bedroom-specific Zillow history includes all Zillow home types, so use it as a bedroom benchmark rather than a pure single-family segment.')
        except (ValidationError, ZillowDataError):
            warnings.append('Bedroom-specific Zillow history was not available, so the comparison uses all single-family homes.')

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
            'relative_to_market_percent': None,
            'market_value': market_value,
        }

    return {
        'price': price,
        'price_range_low': valuation.get('price_range_low'),
        'price_range_high': valuation.get('price_range_high'),
        'source': valuation.get('source'),
        'as_of': valuation.get('as_of') or market.get('latest_date'),
        'relative_to_market_percent': percent_change(price, market_value) if price and market_value else None,
        'market_value': market_value,
    }
