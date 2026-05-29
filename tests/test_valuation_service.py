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
