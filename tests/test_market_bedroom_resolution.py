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


class FakeAddressRentCastClient:
    def value_estimate(self, analysis_request):
        return {
            'subject': {
                'formatted_address': analysis_request['address'],
                'city': 'Austin',
                'state': 'TX',
                'bedrooms': 4,
            },
            'valuation': {'price': 750000, 'source': 'RentCast AVM'},
            'comparables': [],
        }


def test_resolved_subject_bedrooms_select_zillow_segment():
    result = resolve_analysis(
        {
            'address': '100 Main St, Austin, TX',
            'city': None,
            'state': None,
            'location': None,
            'bedrooms': None,
            'period_months': 12,
        },
        market_data=FakeMarketData(),
        rentcast_client=FakeAddressRentCastClient(),
    )

    assert result['subject']['bedrooms'] == 4
    assert result['market']['primary_label'] == '4 bedroom homes'
    assert result['market']['bedroom_latest_value'] == 560000
