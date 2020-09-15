import pytest
import pandas as pd
from ..hydrological_toolbox.data import download_elevation


def test_download_altitude_coordinates():
    result = download_elevation.download_altitude(locations=[[34.05, -81.03]])
    alternative_format = download_elevation.download_altitude(locations=[34.05, -81.03])
    tuple_format = download_elevation.download_altitude(locations=(34.05, -81.03))

    for entry in [result, alternative_format, tuple_format]:
        # non empty entry with three fields
        assert len(entry) == 1
        assert isinstance(entry, pd.DataFrame)
        assert 'LAT' in entry
        assert 'LON' in entry
        assert 'ALTITUDE' in entry


def test_download_altitude_multiple_coords():
    result = download_elevation.download_altitude(locations=[[34.05, -81.03], [34.05, -81.03]])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_download_altitude_address():
    result = download_elevation.download_altitude(locations='Columbia, SC')
    assert isinstance(result, pd.DataFrame)
    assert 'LAT' in result
    assert 'LON' in result
    assert 'ALTITUDE' in result


def test_download_altitude_df():
    charlotte = [35.2271, -80.8431]
    columbia = [34.05, -81.03]
    df = pd.DataFrame([columbia, charlotte])
    df.columns = ['lat', 'lon']

    result = download_elevation.download_altitude(locations=df, lat='lat', lon='lon')
    assert isinstance(result, pd.DataFrame)
    assert 'lat' in result
    assert 'lat' in result
    assert 'ALTITUDE' in result


def test_download_altitude_exception():
    charlotte = [35.2271, -80.8431]
    columbia = [34.05, -81.03]
    df = pd.DataFrame([columbia, charlotte])
    df.columns = ['lat', 'lon']
    with pytest.raises(KeyError):
        download_elevation.download_altitude(locations=df)


