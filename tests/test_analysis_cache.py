import main


def test_analysis_cache_round_trip(monkeypatch):
    monkeypatch.setattr(main, 'ANALYSIS_CACHE_TTL_SECONDS', 60)
    main._ANALYSIS_CACHE.clear()
    analysis_request = {
        'address': '1600 Pennsylvania Avenue NW, Washington, DC 20500',
        'period_months': 120,
    }
    result = {'subject': {'formatted_address': analysis_request['address']}}

    main._cache_analysis(analysis_request, result)

    assert main._get_cached_analysis(analysis_request) == result


def test_disabled_analysis_cache_does_not_store(monkeypatch):
    monkeypatch.setattr(main, 'ANALYSIS_CACHE_TTL_SECONDS', 0)
    main._ANALYSIS_CACHE.clear()

    main._cache_analysis({'address': '123 Main St'}, {'value': 1})

    assert main._ANALYSIS_CACHE == {}
