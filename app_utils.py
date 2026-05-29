FORECAST_LIMITS = {'month': 120, 'year': 10}


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

    return f'{normalized_city.title()}, {normalized_state}'


def percent_change(current_value, reference_value):
    if reference_value == 0:
        raise ValidationError('Cannot compute percent change from a zero reference value.')

    return 100 * (current_value - reference_value) / reference_value


def direction_word(percent_change_value, tense='present'):
    if tense == 'past':
        return 'increased' if percent_change_value > 0 else 'decreased'

    return 'increase' if percent_change_value > 0 else 'decrease'
