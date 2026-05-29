import pandas as pd
import pytest

from app_utils import ValidationError
from zillow_data import bedroom_series_key, load_source_csv, wide_zillow_to_city_series


def sample_zillow_frame():
    return pd.DataFrame({
        'RegionID': [1, 2],
        'SizeRank': [1, 2],
        'RegionName': ['Los Angeles', 'Austin'],
        'RegionType': ['city', 'city'],
        'StateName': ['CA', 'TX'],
        'State': ['CA', 'TX'],
        'Metro': ['Los Angeles-Long Beach-Anaheim, CA', 'Austin-Round Rock-Georgetown, TX'],
        'CountyName': ['Los Angeles County', 'Travis County'],
        '2024-01-31': [900000, 500000],
        '2024-02-29': [910000, 505000],
    })


def test_wide_zillow_to_city_series_extracts_latest_date():
    series, metadata = wide_zillow_to_city_series(sample_zillow_frame(), 'Los Angeles, CA')

    assert list(series.values) == [900000, 910000]
    assert metadata['latest_date'] == '2024-02-29'
    assert metadata['metro'] == 'Los Angeles-Long Beach-Anaheim, CA'


def test_wide_zillow_to_city_series_rejects_missing_city():
    with pytest.raises(ValidationError):
        wide_zillow_to_city_series(sample_zillow_frame(), 'Missing, CA')


def test_bedroom_series_key_uses_five_plus_bucket():
    assert bedroom_series_key(None) is None
    assert bedroom_series_key(0) is None
    assert bedroom_series_key(3) == 'bedroom_3'
    assert bedroom_series_key(7) == 'bedroom_5'


def test_load_source_csv_uses_fresh_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    cached_file = cache_dir / 'sfr.csv'
    cached_file.write_text('RegionName,State,2024-01-31\nAustin,TX,1\n')

    def fail_download(*args, **kwargs):
        raise AssertionError('fresh cache should not download')

    monkeypatch.setenv('HOUSING_CACHE_DIR', str(cache_dir))
    monkeypatch.setenv('HOUSING_DATA_TTL_HOURS', '24')
    monkeypatch.setattr('zillow_data.urlretrieve', fail_download)

    frame, metadata = load_source_csv('sfr')

    assert frame.loc[0, 'RegionName'] == 'Austin'
    assert metadata['cache_path'] == str(cached_file)


def test_load_source_csv_refreshes_stale_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'

    def fake_download(url, path):
        path.write_text('RegionName,State,2024-01-31\nAustin,TX,2\n')

    monkeypatch.setenv('HOUSING_CACHE_DIR', str(cache_dir))
    monkeypatch.setenv('HOUSING_DATA_TTL_HOURS', '0')
    monkeypatch.setattr('zillow_data.urlretrieve', fake_download)

    frame, _ = load_source_csv('sfr')

    assert frame.loc[0, '2024-01-31'] == 2
