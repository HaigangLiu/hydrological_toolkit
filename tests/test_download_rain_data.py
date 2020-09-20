import pandas as pd

from ..hydrological_toolbox.data import download_rain_data


def test_download_rain_coordinates():
    entry = download_rain_data.download_rainfall(start_date='2015-10-01',
                                                 end_date='2015-10-07',
                                                 location=[[34.05, -81.03]])
    assert len(entry) == 7
    assert isinstance(entry, pd.DataFrame)
    assert 'LAT' in entry
    assert 'LON' in entry
    assert 'PRCP' in entry


def test_download_rain_coordinates_df():
    charlotte = [35.2271, -80.8431]
    columbia = [34.05, -81.03]
    df = pd.DataFrame([columbia, charlotte])
    df.columns = ['lat', 'lon']

    result = download_rain_data.download_rainfall(location=df,
                                                  start_date='2014-01-01',
                                                  end_date='2014-01-02',
                                                  lat='lat',
                                                  lon='lon')
    assert isinstance(result, pd.DataFrame)
    assert 'PRCP' in result


def test_download_rain_coordinates_state_name():
    result = download_rain_data.download_rainfall(location='SC',
                                                  start_date='2015-10-01',
                                                  end_date='2015-10-10')
    assert len(result) > 0
    assert isinstance(result, pd.DataFrame)
    assert result.PRCP.sum() > 0  # the flood event period in south carolina


def test_download_rain_coordinates_by_address():
    result = download_rain_data.download_rainfall(location='800 State Street, West Columbia, SC',
                                                  start_date='2015-10-01',
                                                  end_date='2015-10-10')
    assert len(result) == 10
    assert isinstance(result, pd.DataFrame)
    assert result.PRCP.sum() > 0  # the flood event period in south carolina

