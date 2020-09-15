import pytest
import pandas as pd
from ..hydrological_toolbox.data import download_flood_data


def test_download_flood():
    result = download_flood_data.download_flood(start_date='2016-12-12',
                                                end_date='2016-12-31',
                                                locations=[[32, -82]])
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'LAT' in result and 'LON' in result


def test_download_flood_no_mans_land():
    """
    if the coordinates is not the United States at all,
    or there is no locations nearby, then the download function will
    raise a value error.
    """
    with pytest.raises(ValueError):
        download_flood_data.download_flood(start_date='2016-12-12',
                                           end_date='2016-12-31',
                                           locations=[[70, 82]])


def test_download_flood_state_name():
    result = download_flood_data.download_flood(start_date='2016-12-12',
                                                end_date='2016-12-31',
                                                locations='SC')
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


def test_download_flood_site_number():
    result = download_flood_data.download_flood(start_date='2016-12-12',
                                                end_date='2016-12-31',
                                                locations=['02110400'])
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


def test_download_flood_bounding_box():
    result = download_flood_data.download_flood(start_date='2016-12-12',
                                                end_date='2016-12-31',
                                                lat_max=32,
                                                lat_min=30,
                                                lon_max=-81,
                                                lon_min=-82)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


