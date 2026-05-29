import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd

from app_utils import ValidationError


ZILLOW_BASE_URL = 'https://files.zillowstatic.com/research/public_csvs/zhvi'
DEFAULT_SERIES_URLS = {
    'sfr': f'{ZILLOW_BASE_URL}/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv',
    'bedroom_1': f'{ZILLOW_BASE_URL}/City_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'bedroom_2': f'{ZILLOW_BASE_URL}/City_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'bedroom_3': f'{ZILLOW_BASE_URL}/City_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'bedroom_4': f'{ZILLOW_BASE_URL}/City_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'bedroom_5': f'{ZILLOW_BASE_URL}/City_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}


class ZillowDataError(RuntimeError):
    pass


def _cache_dir():
    return Path(os.getenv('HOUSING_CACHE_DIR', Path(__file__).parent / '.cache' / 'housing'))


def _ttl_seconds():
    raw_ttl = os.getenv('HOUSING_DATA_TTL_HOURS', '24')
    try:
        ttl_hours = float(raw_ttl)
    except ValueError:
        ttl_hours = 24
    return max(0, ttl_hours) * 3600


def _series_url(series_key):
    env_key = f'ZILLOW_{series_key.upper()}_URL'
    return os.getenv(env_key, DEFAULT_SERIES_URLS[series_key])


def _cache_path(series_key):
    return _cache_dir() / f'{series_key}.csv'


def _is_fresh(path, ttl_seconds):
    if not path.exists():
        return False
    if ttl_seconds == 0:
        return False
    return (time.time() - path.stat().st_mtime) < ttl_seconds


def load_source_csv(series_key):
    path = _cache_path(series_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = _series_url(series_key)

    if not _is_fresh(path, _ttl_seconds()):
        try:
            urlretrieve(url, path)
        except (OSError, URLError) as exc:
            if not path.exists():
                raise ZillowDataError('Zillow housing data could not be loaded right now.') from exc

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ZillowDataError('Cached Zillow housing data could not be read.') from exc

    return frame, {'source_url': url, 'cache_path': str(path), 'fetched_at': path.stat().st_mtime}


def wide_zillow_to_city_series(frame, location):
    required_columns = {'RegionName', 'State'}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ZillowDataError('Zillow data did not contain the expected city/state columns.')

    rows = frame[
        (frame['RegionName'].astype(str).str.lower() == location.split(', ')[0].lower())
        & (frame['State'].astype(str).str.upper() == location.split(', ')[1].upper())
    ]
    if rows.empty:
        raise ValidationError(f'{location} is either misspelled or not available in the Zillow city data.')

    row = rows.iloc[0]
    date_columns = [column for column in frame.columns if _looks_like_date_column(column)]
    series = pd.Series(row[date_columns].values, index=pd.to_datetime(date_columns), name=location)
    series = pd.to_numeric(series, errors='coerce').dropna().sort_index()
    if series.empty:
        raise ZillowDataError(f'Zillow did not return usable time-series values for {location}.')

    metadata = {
        'region_name': row.get('RegionName'),
        'state': row.get('State'),
        'metro': row.get('Metro'),
        'county_name': row.get('CountyName'),
        'region_id': row.get('RegionID'),
        'latest_date': series.index[-1].date().isoformat(),
    }
    return series, metadata


def _looks_like_date_column(column):
    text = str(column)
    return len(text) == 10 and text[4] == '-' and text[7] == '-'


def bedroom_series_key(bedrooms):
    if bedrooms is None:
        return None
    if bedrooms >= 5:
        return 'bedroom_5'
    if bedrooms >= 1:
        return f'bedroom_{int(bedrooms)}'
    return None


class ZillowMarketData:
    def get_city_series(self, location, series_key='sfr'):
        frame, source_metadata = load_source_csv(series_key)
        series, city_metadata = wide_zillow_to_city_series(frame, location)
        city_metadata.update(source_metadata)
        city_metadata['series_key'] = series_key
        return series, city_metadata
