import pandas as pd

from valuation_service import resolve_analysis


class FakeMarketData:
    def get_city_series(self, location, series_key='sfr'):
        values = [500000, 525000] if series_key == 'sfr' else [540000, 560000]
        series = pd.Series(
            values,
            index=pd.to_datetime(['2024-01-31', '2024-02-29']),
            name=location,
        )
        return series, {'series_key': series_key, 'latest_date': '2024-02-29'}


class FakeRentCastClient:
    def value_estimate(self, analysis_request):
        raise AssertionError('manual market benchmark should not call RentCast')


class FakeAddressRentCastClient:
    def value_estimate(self, analysis_request):
        return {
            'valuation': {'price': 560000, 'price_range_low': 530000, 'price_range_high': 590000, 'source': 'RentCast AVM'},
            'subject': {
                'formatted_address': analysis_request['address'],
                'city': 'Austin',
                'state': 'TX',
                'bedrooms': 3,
                'square_footage': 1800,
            },
            'comparables': [],
        }

    def active_sale_listing(self, analysis_request):
        return {
            'status': 'Active',
            'price': 525000,
            'listed_date': '2026-08-01T00:00:00.000Z',
            'days_on_market': 5,
        }


def test_resolve_analysis_uses_market_benchmark_without_address():
    result = resolve_analysis(
        {
            'address': None,
            'city': 'Austin',
            'state': 'TX',
            'location': 'Austin, TX',
            'bedrooms': 3,
            'period_months': 12,
        },
        market_data=FakeMarketData(),
        rentcast_client=FakeRentCastClient(),
    )

    assert result['valuation']['price'] == 560000
    assert result['valuation']['source'] == 'Zillow market benchmark'
    assert result['market']['primary_label'] == '3 bedroom homes'
    assert result['warnings'][0].startswith('No address was supplied')


def test_resolve_analysis_includes_active_subject_listing():
    result = resolve_analysis(
        {
            'address': '123 Main St, Austin, TX 78701',
            'city': None,
            'state': None,
            'location': None,
            'period_months': 12,
        },
        market_data=FakeMarketData(),
        rentcast_client=FakeAddressRentCastClient(),
    )

    assert result['subject']['listing_status'] == 'Active'
    assert result['subject']['listing_price'] == 525000
    assert result['subject']['days_on_market'] == 5
