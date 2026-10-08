import pytest

from housing_safety import initialize_database


@pytest.fixture(autouse=True)
def isolated_safety_ledger(monkeypatch, tmp_path_factory):
    """Every test uses a real disposable ledger, never the production allowance."""
    path = tmp_path_factory.mktemp('admission') / 'ledger.sqlite3'
    initialize_database(path)
    monkeypatch.setenv('HOUSING_SAFETY_DB_PATH', str(path))
    monkeypatch.setenv('HOUSING_RENTCAST_MAX_REQUESTS_31D', '1000')
    monkeypatch.setenv('HOUSING_VISITOR_REQUESTS_PER_MINUTE', '1000')
    monkeypatch.setenv('HOUSING_VISITOR_REQUESTS_PER_DAY', '1000')
    monkeypatch.setenv('HOUSING_GLOBAL_REQUESTS_PER_MINUTE', '1000')
    import main
    main._ANALYSIS_CACHE.clear()
    yield path
    main._ANALYSIS_CACHE.clear()
