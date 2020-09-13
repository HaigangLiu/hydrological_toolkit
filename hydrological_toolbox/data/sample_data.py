from pkg_resources import resource_stream
from typing import Union
import logging
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

from data.util import convert_daily_to_monthly, spatial_left_join_k_neighbors

logger = logging.getLogger(__name__)


def load_daily_rain(description: bool = True) -> pd.DataFrame:
    """
    This function is going to print a description of the dataset. set description=False to silent this.
    """
    description_of_daily_rain = """
    This is the daily rainfall from 2015 in South Carolina from https://water.weather.gov/precip/download.php
    * The daily data is accumulation in inches.
    * Here's a short intro according to National Weather Service:
    ```
    The precipitation data are quality-controlled, multi-sensor (radar and rain gauge) precipitation estimates 
    obtained from National Weather Service (NWS) River Forecast Centers (RFCs) and mosaicked by
    National Centers for Environmental Prediction (NCEP). 
    ```
    """
    if description:
        logger.critical(description_of_daily_rain)
    file_ = resource_stream('asset.sample_data_sc', 'sc_rain_2015.parquet')
    return pd.read_parquet(file_)


def load_daily_flood(description: bool = True) -> pd.DataFrame:
    description_of_flood = """
    This is the daily gage level from USGS. For further information once can refer to this link
     https://waterdata.usgs.gov/nwis

    Here's a brief intro:
    These pages provide access to water-resources data collected at approximately
    1.9 million sites in all 50 States, the District of Columbia and etc.

    Theses sites measure different aspect of water resource data:
    The USGS investigates the occurrence, quantity, quality, distribution, and
    movement of surface and underground waters. We focus on surface water and gage level measurement.
    Gage level are measured in feet.
    """

    if description:
        logger.critical(description_of_flood)

    daily_flood_file = resource_stream('asset.sample_data_sc', 'GAGE-20110101-20161231-SC.parquet')
    daily_flood = pd.read_parquet(daily_flood_file)
    daily_flood['DATE'] = pd.to_datetime(daily_flood.DATE)
    daily_flood = daily_flood[daily_flood.DATE.dt.year == 2015]
    return daily_flood


def load_monthly_rain(agg_function: Union[None, dict] = None, description: bool = True) -> pd.DataFrame:
    """
    This function is going to print a description of the dataset. set description=False to silent this.
    """
    description_of_monthly_rain = """
    * This is the monthly aggregation of daily rain, whose description can be found by calling 
    load_daily_rain(description=True).
    * The aggregation function is to get sum/mean/max/min of the daily accumulation data by default.
    User can change the default by setting for example
    `agg_funcs = {value_col: [('MIN', np.min), ('MAX', np.max)]}` to get only max and min of daily value for each month.
    * The unit is inch.
    """
    if description:
        logger.critical(description_of_monthly_rain)

    daily_rain = load_daily_rain(description=False)
    monthly_rain = convert_daily_to_monthly(df=daily_rain,
                                            coordinate_x='LAT',
                                            coordinate_y='LON',
                                            date_col='DATE',
                                            value_col='PRCP',
                                            agg_funcs=agg_function)  # use the default agg_func
    return monthly_rain


def load_monthly_flood(daily_source='GAGE_MEAN',
                       agg_funcs=None,
                       description: bool = True) -> pd.DataFrame:
    """
    User can choose what kind of daily info to aggregate to get the monthly data
    default choice is daily mean. In other words we get monthly max, min, and mean
    out of daily mean. However this setting can be altered by setting daily resource='GAGE_MIN'
    for example to use daily minimum.
    """

    monthly_flood_description = """
    * This is the monthly aggregation of daily flood, whose description can be found by calling 
    load_daily_flood(description=True).
    * The aggregation function is to get sum/mean/max/min of the daily MEAN data by default.
    User can change the default by setting for example
    `agg_funcs = {value_col: [('MIN', np.min), ('MAX', np.max)]}` to get only max and min of daily value for each month.
    Additionally, if one wish to get the monthly aggregation of daily max gage rather than average 
    gage level, set daily_source 'GAGE_MAX' when calling this function.
    * The unit is feet.
    """
    if description:
        logger.critical(monthly_flood_description)

    available_daily_sources = ['GAGE_MAX', 'GAGE_MIN', 'GAGE_MEAN']

    if daily_source not in available_daily_sources:
        raise ValueError(f'This method only support deriving monthly statistics from {available_daily_sources}')

    logger.critical('This is the monthly flood/gage data from 2015 in South Carolina')
    daily_flood = load_daily_flood(description=False)
    # using the site number as the id column will discard lat and lon.
    # we need to merge it back to get the lat and lon
    ref_table = daily_flood[['SITENUMBER', 'LAT', 'LON']]

    monthly_flood = convert_daily_to_monthly(df=daily_flood,
                                             coordinate_x='SITENUMBER',
                                             date_col='DATE',
                                             value_col=daily_source,
                                             agg_funcs=agg_funcs)

    monthly_flood = pd.merge(left=monthly_flood,
                             right=ref_table,
                             how='inner',
                             on=['SITENUMBER'])

    rename_dict = {}
    # the original name is, for example, GAGE_MIN_MAX, means the monthly max of daily min
    # however this is confusion in general so we will simplify it to GAGE_MAX
    for col in monthly_flood.columns:
        if col.startswith('GAGE'):
            words_in_col = col.split('_')
            gage, *b, stats = words_in_col
            rename_dict[col] = '_'.join([gage, stats])

    monthly_flood.drop_duplicates(inplace=True)
    monthly_flood.rename(rename_dict, inplace=True, axis=1)
    return monthly_flood


def load_monthly_flood_and_rain(description: bool = True) -> pd.DataFrame:
    description_for_flood_and_rain = """
    This is the data for rainfall amount (inch) and gage level (feet) combined during 
    2015. The data is obtained by aggregating the daily records.
    
    Note that the way to combine is not merging exactly based on keys, but finding 
    the nearest rainfall value for each flood location. This is because the rainfall 
    data are more dense than the flood measurement sites.
    """
    if description:
        logger.critical(description_for_flood_and_rain)

    month_flood = load_monthly_flood(description=False)
    month_rain = load_monthly_rain(description=False)

    result = []
    for month in range(1, 13):
        rain = month_rain[(month_rain.MONTH == month) & (month_rain.YEAR == 2015)]
        flood = month_flood[(month_flood.MONTH == month) & (month_flood.YEAR == 2015)]
        result.append(spatial_left_join_k_neighbors(flood, rain, left_on=['LAT', 'LON'], right_on=['LAT', 'LON'], k=1))

    output = pd.concat(result)
    logger.critical(output.columns)
    output.rename({'YEAR_x': 'YEAR',
                   'MONTH_x': 'MONTH',
                   'LAT': 'LAT_GAGE',
                   'LON': 'LON_GAGE',
                   'LAT_RIGHT_0': 'LAT_PRCP',
                   'LON_RIGHT_0': 'LON_PRCP',
                   'PRCP_MEAN_RIGHT_0': 'PRCP_MEAN',
                   'PRCP_MAX_RIGHT_0': 'PRCP_MAX',
                   'PRCP_MIN_RIGHT_0': 'PRCP_MIN',
                   'PRCP_SUM_RIGHT_0': 'PRCP_SUM',
                   },
                  axis=1,
                  inplace=True)

    columns_kept = ['SITENUMBER', 'YEAR', 'MONTH', 'LAT_GAGE', 'LON_GAGE', 'LAT_PRCP', 'LON_PRCP',
                    'GAGE_MIN', 'GAGE_MAX', 'GAGE_MEAN', 'GAGE_SUM',
                    'PRCP_MIN', 'PRCP_MAX', 'PRCP_MEAN', 'PRCP_SUM']

    return output[columns_kept]


def load_daily_flood_and_rain(description: bool = True):
    """
    This is the flood and rainfall data in South Carolina (daily)
    """
    description_for_flood_and_rain = """
    This is the daily data for rainfall amount (inch) and gage level (feet) combined during 
    2015 from Jan 1 to Dec 31. 

    Note that the way to combine is not merging exactly based on keys, but finding 
    the nearest rainfall value for each flood location. This is because the rainfall 
    data are more dense than the flood measurement sites.
    """
    if description:
        logger.critical(description_for_flood_and_rain)

    locations_in_flood = load_daily_flood(description=False)[['LAT', 'LON']].drop_duplicates()
    locations_in_flood['LAT'] = locations_in_flood['LAT'].astype(float)
    locations_in_flood['LON'] = locations_in_flood['LON'].astype(float)

    locations_in_rain = load_daily_rain(description=False)[['LAT', 'LON']].drop_duplicates()
    locations_in_rain['LAT'] = locations_in_rain['LAT'].astype(float)
    locations_in_rain['LON'] = locations_in_rain['LON'].astype(float)

    distance_ = distance_matrix(locations_in_flood[['LAT', 'LON']], locations_in_rain[['LAT', 'LON']].values)
    min_index_for_each_flood_location = np.argmin(distance_, axis=1)

    relevant_rain_fall_stations = locations_in_rain.iloc[min_index_for_each_flood_location].reset_index(drop=True)
    locations_in_flood = locations_in_flood.reset_index(drop=True)
    combined_rain_and_flood = pd.concat([locations_in_flood, relevant_rain_fall_stations], axis=1)
    combined_rain_and_flood.columns = ['LAT_GAGE', 'LON_GAGE', 'LAT_PRCP', 'LON_PRCP']

    all_flood_data = load_daily_flood(description=False)
    all_flood_data['LAT'] = all_flood_data['LAT'].astype(float)
    all_flood_data['LON'] = all_flood_data['LON'].astype(float)

    flood_with_ref_table = pd.merge(all_flood_data,
                                    combined_rain_and_flood,
                                    left_on=['LAT', 'LON'],
                                    right_on=['LAT_GAGE', 'LON_GAGE'],
                                    how='left')

    all_rain = load_daily_rain(description=False)
    all_rain['LAT'] = all_rain['LAT'].astype(float)
    all_rain['LON'] = all_rain['LON'].astype(float)
    all_rain['DATE'] = pd.to_datetime(all_rain.DATE)
    output = pd.merge(left=flood_with_ref_table,
                      right=all_rain,
                      left_on=['LAT_PRCP', 'LON_PRCP', 'DATE'],
                      right_on=['LAT', 'LON', 'DATE'],
                      how='left').drop_duplicates()
    output.drop(columns=['LAT_y', 'LON_y', 'LAT_x', 'LON_x'], inplace=True)
    output = output[['SITENUMBER', 'STATION_NAME', 'LAT_GAGE', 'LON_GAGE', 'LAT_PRCP', 'LON_PRCP',
                     'STATE', 'DATE', 'GAGE_MAX', 'GAGE_MIN', 'GAGE_MEAN', 'PRCP']]
    return output

