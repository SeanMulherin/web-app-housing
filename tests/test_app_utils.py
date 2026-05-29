import pytest

from app_utils import (
    ValidationError,
    direction_word,
    normalize_location,
    percent_change,
    period_to_months,
    validate_forecast_period,
)


def test_validate_forecast_period_accepts_months_and_years():
    assert validate_forecast_period('10', 'year') == 10
    assert validate_forecast_period('120', 'month') == 120
    assert period_to_months(2, 'year') == 24


def test_validate_forecast_period_rejects_invalid_values():
    with pytest.raises(ValidationError):
        validate_forecast_period('0', 'month')
    with pytest.raises(ValidationError):
        validate_forecast_period('1.5', 'year')
    with pytest.raises(ValidationError):
        validate_forecast_period('11', 'year')
    with pytest.raises(ValidationError):
        validate_forecast_period('5', 'day')


def test_normalize_location_requires_city_and_state():
    assert normalize_location('los angeles', 'ca') == 'Los Angeles, CA'

    with pytest.raises(ValidationError):
        normalize_location('', 'CA')
    with pytest.raises(ValidationError):
        normalize_location('Los Angeles', '')


def test_percent_change_and_direction_words_are_independent():
    assert percent_change(110, 100) == 10
    assert direction_word(10, tense='present') == 'increase'
    assert direction_word(-10, tense='present') == 'decrease'
    assert direction_word(10, tense='past') == 'increased'
    assert direction_word(-10, tense='past') == 'decreased'
