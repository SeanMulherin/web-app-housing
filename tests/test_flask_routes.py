import pandas as pd


def sample_analysis(warnings=None):
    series = pd.Series(
        [500000, 525000],
        index=pd.to_datetime(['2024-01-31', '2024-02-29']),
        name='Austin, TX',
    )
    return {
        'subject': {
            'formatted_address': '123 Main St, Austin, TX 78701',
            'city': 'Austin',
            'state': 'TX',
            'zip_code': '78701',
            'property_type': 'Single Family',
            'bedrooms': 3,
            'bathrooms': 2,
            'square_footage': 1800,
            'lot_size': 7000,
            'year_built': 1990,
            'last_sale_price': 450000,
            'last_sale_date': '2021-01-01',
        },
        'valuation': {
            'price': 550000,
            'price_range_low': 520000,
            'price_range_high': 580000,
            'source': 'RentCast AVM',
            'as_of': '2026-05-29',
            'relative_to_market_percent': 4.8,
            'market_value': 525000,
        },
        'market': {
            'location': 'Austin, TX',
            'sfr_series': series,
            'bedroom_series': None,
            'primary_series': series,
            'primary_label': 'Single-family homes',
            'latest_value': 525000,
            'latest_date': '2024-02-29',
            'lookback_percent_change': 5.0,
        },
        'comparables': [
            {
                'formatted_address': '125 Main St, Austin, TX 78701',
                'status': 'Active',
                'price': 560000,
                'bedrooms': 3,
                'bathrooms': 2,
                'square_footage': 1750,
                'distance': 0.2,
                'correlation': 0.98,
            },
            {
                'formatted_address': '127 Main St, Austin, TX 78701',
                'status': 'Inactive',
                'price': 530000,
                'bedrooms': 3,
                'bathrooms': 2,
                'square_footage': 1800,
                'distance': 0.3,
                'correlation': 0.95,
            }
        ],
        'warnings': warnings or [],
    }


def test_forecast_route_renders_address_valuation(monkeypatch):
    import main

    monkeypatch.setattr(main, 'resolve_analysis', lambda analysis_request: sample_analysis())
    monkeypatch.setattr(main, 'market_history_plot', lambda analysis: '<img alt="market" />')
    monkeypatch.setattr(main, 'comparable_values_plot', lambda analysis: '<img alt="comps" />')
    monkeypatch.setattr(main, 'forecast_plot', lambda analysis, period_months: None)

    client = main.app.test_client()
    response = client.post('/forecast', data={
        'address': '123 Main St, Austin, TX 78701',
        'period': '10',
        'period_unit': 'year',
    })

    assert response.status_code == 200
    assert b'$550,000' in response.data
    assert b'Comparable Properties' in response.data
    assert b'alt="comps"' in response.data
    assert b'125 Main St' in response.data


def test_forecast_route_renders_market_only_warning(monkeypatch):
    import main

    monkeypatch.setattr(
        main,
        'resolve_analysis',
        lambda analysis_request: sample_analysis(['No address was supplied, so the app is showing a market benchmark instead of an address-level AVM.']),
    )
    monkeypatch.setattr(main, 'market_history_plot', lambda analysis: '<img alt="market" />')
    monkeypatch.setattr(main, 'comparable_values_plot', lambda analysis: '<img alt="comps" />')
    monkeypatch.setattr(main, 'forecast_plot', lambda analysis, period_months: None)

    client = main.app.test_client()
    response = client.post('/forecast', data={
        'city': 'Austin',
        'state': 'TX',
        'period': '10',
        'period_unit': 'year',
    })

    assert response.status_code == 200
    assert b'No address was supplied' in response.data


def test_forecast_route_renders_validation_errors():
    import main

    client = main.app.test_client()
    response = client.post('/forecast', data={
        'city': 'Austin',
        'state': 'XX',
        'period': '10',
        'period_unit': 'year',
    })

    assert response.status_code == 200
    assert b'ERROR:' in response.data


def test_analysis_api_serializes_address_result(monkeypatch):
    import main

    monkeypatch.setattr(main, 'resolve_analysis', lambda analysis_request: sample_analysis())

    client = main.app.test_client()
    response = client.post(
        '/api/analysis',
        json={
            'address': '123 Main St, Austin, TX 78701',
            'period': '10',
            'period_unit': 'year',
        },
        headers={'Origin': 'https://housing-market-lab.sean-mulherin.chatgpt.site'},
    )

    assert response.status_code == 200
    assert response.headers['Access-Control-Allow-Origin'] == 'https://housing-market-lab.sean-mulherin.chatgpt.site'
    assert response.json['valuation']['price'] == 550000
    assert response.json['comparables'][0]['formatted_address'].startswith('125 Main St')
    assert response.json['market']['sfr_series'] == [
        {'date': '2024-01-31', 'value': 500000.0},
        {'date': '2024-02-29', 'value': 525000.0},
    ]


def test_analysis_api_returns_json_validation_error():
    import main

    client = main.app.test_client()
    response = client.post('/api/analysis', json={
        'city': 'Austin',
        'state': 'XX',
        'period': '10',
        'period_unit': 'year',
    })

    assert response.status_code == 400
    assert 'error' in response.json


def test_comparable_values_plot_renders_image_without_comparables():
    import main

    analysis = sample_analysis()
    analysis['comparables'] = []

    assert main.comparable_values_plot(analysis) is None


def test_comparable_values_plot_renders_image_with_comparables():
    import main

    plot_html = main.comparable_values_plot(sample_analysis())

    assert plot_html.startswith('<img src="data:image/png;base64,')
    assert 'Comparable properties compared with subject estimate' in plot_html
