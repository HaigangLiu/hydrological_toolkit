import pandas as pd
from ..hydrological_toolbox.data import sample_data


def test_load_daily_rain():
    df = sample_data.load_daily_rain()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'LAT' in df and 'LON' in df and 'PRCP' in df and 'DATE' in df


def test_load_daily_flood():
    df = sample_data.load_daily_flood()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'LAT' in df and 'LON' in df and 'DATE' in df
    assert 'GAGE_MIN' in df and 'GAGE_MAX' in df


def test_load_monthly_rain():
    df = sample_data.load_monthly_rain()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'LON' in df and 'LAT' in df and 'YEAR' in df and 'MONTH' in df
    assert 'PRCP_MAX' in df and 'PRCP_MIN' in df and 'PRCP_SUM' in df


def test_load_monthly_flood():
    df = sample_data.load_monthly_flood()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'LON' in df and 'LAT' in df and 'YEAR' in df and 'MONTH' in df
    assert 'GAGE_MAX' in df and 'GAGE_MEAN' in df and 'GAGE_MIN' in df


def test_load_monthly_rain_and_flood():
    df = sample_data.load_monthly_flood_and_rain()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'LAT_GAGE' in df and 'LON_GAGE' in df
    assert 'LAT_PRCP' in df and 'LON_PRCP' in df
    assert 'PRCP_MEAN' in df and 'GAGE_MAX' in df
    assert 'YEAR' in df and 'MONTH' in df


def test_load_daily_rain_and_flood():
    df = sample_data.load_daily_flood_and_rain()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'PRCP' in df and 'GAGE_MAX' in df
    assert 'LAT_GAGE' in df and 'LON_GAGE' in df and 'LAT_PRCP' in df and 'LON_PRCP' in df
    assert 'DATE' in df


def test_load_sst_daily():
    df = sample_data.load_daily_sst()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'SST' in df and 'LAT' in df and 'LON' in df


def test_load_sst_monthly():
    df = sample_data.load_monthly_sst()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.MONTH.max() == 12
    assert 'SST_MEAN' in df and 'SST_MAX' in df
    assert 'SST_MIN' in df and 'SST_SUM' in df
