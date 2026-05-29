import re


FORECAST_LIMITS = {'month': 120, 'year': 10}
SQFT_PER_ACRE = 43560

STATE_ABBREVIATIONS = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC',
}


class ValidationError(ValueError):
    pass


def parse_positive_integer(raw_value, field_name):
    try:
        numeric_value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f'{field_name} must be a positive whole number.')

    if not numeric_value.is_integer():
        raise ValidationError(f'{field_name} must be a positive whole number.')

    integer_value = int(numeric_value)
    if integer_value <= 0:
        raise ValidationError(f'{field_name} must be greater than 0.')

    return integer_value


def parse_optional_float(raw_value, field_name, minimum=None, maximum=None):
    raw_text = str(raw_value or '').strip()
    if not raw_text:
        return None

    try:
        numeric_value = float(raw_text)
    except ValueError:
        raise ValidationError(f'{field_name} must be a number.')

    if minimum is not None and numeric_value < minimum:
        raise ValidationError(f'{field_name} must be at least {minimum}.')
    if maximum is not None and numeric_value > maximum:
        raise ValidationError(f'{field_name} cannot exceed {maximum}.')

    return numeric_value


def parse_optional_integer(raw_value, field_name, minimum=None, maximum=None):
    raw_text = str(raw_value or '').strip()
    if not raw_text:
        return None

    try:
        numeric_value = float(raw_text)
    except ValueError:
        raise ValidationError(f'{field_name} must be a whole number.')

    if not numeric_value.is_integer():
        raise ValidationError(f'{field_name} must be a whole number.')

    integer_value = int(numeric_value)
    if minimum is not None and integer_value < minimum:
        raise ValidationError(f'{field_name} must be at least {minimum}.')
    if maximum is not None and integer_value > maximum:
        raise ValidationError(f'{field_name} cannot exceed {maximum}.')

    return integer_value


def validate_forecast_period(raw_period, period_unit):
    if period_unit not in FORECAST_LIMITS:
        raise ValidationError('Forecast unit must be month or year.')

    period = parse_positive_integer(raw_period, 'Forecast length')
    max_period = FORECAST_LIMITS[period_unit]
    if period > max_period:
        raise ValidationError(f'Forecast length cannot exceed {max_period} {period_unit}(s).')

    return period


def period_to_months(period, period_unit):
    if period_unit == 'year':
        return period * 12
    return period


def normalize_location(city, state):
    normalized_city = str(city).strip()
    normalized_state = str(state).strip().upper()

    if not normalized_city:
        raise ValidationError('City is required.')
    if not normalized_state:
        raise ValidationError('State is required.')
    if normalized_state not in STATE_ABBREVIATIONS:
        raise ValidationError('State must be a valid U.S. postal abbreviation.')

    return f'{normalized_city.title()}, {normalized_state}'


def city_state_from_address(address):
    parts = [part.strip() for part in str(address or '').split(',') if part.strip()]
    if len(parts) < 3:
        return None, None

    city = parts[-2]
    state_match = re.search(r'\b([A-Za-z]{2})\b', parts[-1])
    if not state_match:
        return None, None

    state = state_match.group(1).upper()
    if state not in STATE_ABBREVIATIONS:
        return None, None

    return city, state


def normalize_analysis_request(form):
    address = str(form.get('address', '') or '').strip()
    city = str(form.get('city', '') or '').strip()
    state = str(form.get('state', '') or '').strip().upper()
    period_unit = form.get('period_unit', '')
    period = validate_forecast_period(form.get('period', ''), period_unit)

    if not address:
        location = normalize_location(city, state)
    else:
        if not city and not state:
            city, state = city_state_from_address(address)
        location = normalize_location(city, state) if city or state else None

    bedrooms = parse_optional_integer(form.get('bedrooms'), 'Bedrooms', minimum=0, maximum=20)
    bathrooms = parse_optional_float(form.get('bathrooms'), 'Bathrooms', minimum=0, maximum=20)
    square_footage = parse_optional_float(form.get('square_footage'), 'Square footage', minimum=100, maximum=50000)
    acres = parse_optional_float(form.get('acres'), 'Acres', minimum=0, maximum=1000)
    year_built = parse_optional_integer(form.get('year_built'), 'Year built', minimum=1700, maximum=2100)

    return {
        'address': address or None,
        'city': city.title() if city else None,
        'state': state or None,
        'location': location,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'square_footage': square_footage,
        'acres': acres,
        'lot_size': acres * SQFT_PER_ACRE if acres is not None else None,
        'year_built': year_built,
        'property_type': 'Single Family',
        'period': period,
        'period_unit': period_unit,
        'period_months': period_to_months(period, period_unit),
    }


def percent_change(current_value, reference_value):
    if reference_value == 0:
        raise ValidationError('Cannot compute percent change from a zero reference value.')

    return 100 * (current_value - reference_value) / reference_value


def direction_word(percent_change_value, tense='present'):
    if tense == 'past':
        return 'increased' if percent_change_value > 0 else 'decreased'

    return 'increase' if percent_change_value > 0 else 'decrease'
