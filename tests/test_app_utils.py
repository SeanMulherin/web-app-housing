import pytest

from app_utils import (
    SQFT_PER_ACRE,
    ValidationError,
    direction_word,
    normalize_analysis_request,
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
    with pytest.raises(ValidationError):
        normalize_location('Los Angeles', 'XX')


def test_normalize_analysis_request_accepts_address_only():
    result = normalize_analysis_request({
        'address': '5500 Grand Lake Dr, San Antonio, TX 78244',
        'period': '6',
        'period_unit': 'month',
    })

    assert result['address'] == '5500 Grand Lake Dr, San Antonio, TX 78244'
    assert result['city'] == 'San Antonio'
    assert result['state'] == 'TX'
    assert result['location'] == 'San Antonio, TX'
    assert result['property_type'] == 'Single Family'
    assert result['period_months'] == 6


def test_normalize_analysis_request_accepts_manual_characteristics():
    result = normalize_analysis_request({
        'city': 'los angeles',
        'state': 'ca',
        'bedrooms': '3',
        'bathrooms': '2.5',
        'square_footage': '2200',
        'acres': '0.25',
        'year_built': '1985',
        'period': '10',
        'period_unit': 'year',
    })

    assert result['location'] == 'Los Angeles, CA'
    assert result['bedrooms'] == 3
    assert result['bathrooms'] == 2.5
    assert result['square_footage'] == 2200
    assert result['lot_size'] == 0.25 * SQFT_PER_ACRE
    assert result['year_built'] == 1985
    assert result['period_months'] == 120


def test_normalize_analysis_request_rejects_invalid_characteristics():
    with pytest.raises(ValidationError):
        normalize_analysis_request({
            'city': 'Los Angeles',
            'state': 'CA',
            'square_footage': 'tiny',
            'period': '10',
            'period_unit': 'year',
        })

    with pytest.raises(ValidationError):
        normalize_analysis_request({
            'city': 'Los Angeles',
            'state': 'CA',
            'bedrooms': '2.5',
            'period': '10',
            'period_unit': 'year',
        })


def test_percent_change_and_direction_words_are_independent():
    assert percent_change(110, 100) == 10
    assert direction_word(10, tense='present') == 'increase'
    assert direction_word(-10, tense='present') == 'decrease'
    assert direction_word(10, tense='past') == 'increased'
    assert direction_word(-10, tense='past') == 'decreased'
