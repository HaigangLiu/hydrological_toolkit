import pandas as pd

from ..hydrological_toolbox.data import download_hydrological_data


def test_download_flood():
    result = download_hydrological_data.download_flood(start_date='2016-12-12',
                                                       end_date='2016-12-31',
                                                       locations=[[32, -82]])
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'LAT' in result and 'LON' in result


def test_download_flood_no_mans_land():
    """
    if the coordinates is not the United States at all,
    or there is no locations nearby,
    then this function will return None and give a warning
    """
    f = download_hydrological_data.download_flood(start_date='2016-12-12',
                                                  end_date='2016-12-31',
                                                  locations=[[70, 82]])
    assert f is None


def test_download_flood_state_name():
    result = download_hydrological_data.download_flood(start_date='2016-12-12',
                                                       end_date='2016-12-31',
                                                       locations='SC')
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


def test_download_flood_site_number():
    result = download_hydrological_data.download_flood(start_date='2016-12-12',
                                                       end_date='2016-12-31',
                                                       locations=['02110400'])
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


def test_download_flood_bounding_box():
    result = download_hydrological_data.download_flood(start_date='2016-12-12',
                                                       end_date='2016-12-31',
                                                       lat_max=32,
                                                       lat_min=30,
                                                       lon_max=-81,
                                                       lon_min=-82)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'GAGE_MEAN' in result


def test_download_other_variables_bounding_box():
    result = download_hydrological_data.download_hydrological_variable(start_date='2016-12-12',
                                                                       end_date='2016-12-31',
                                                                       variable_name='DISCHARGE',
                                                                       variable_id='00060',
                                                                       lat_max=32,
                                                                       lat_min=30,
                                                                       lon_max=-81,
                                                                       lon_min=-82)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'DISCHARGE_MEAN' in result


def test_download_other_variables_by_state_name():
    result = download_hydrological_data.download_hydrological_variable(start_date='2016-12-12',
                                                                       end_date='2016-12-31',
                                                                       variable_name='PRCP',
                                                                       variable_id='00045',
                                                                       locations='SC')
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 1
    assert 'PRCP_SUM' in result
