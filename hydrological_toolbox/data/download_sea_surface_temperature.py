from contextlib import closing
from datetime import date, timedelta, datetime
import logging
import os
import shutil
from typing import Union
import urllib.request as request
import tempfile

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from data.util import convert_coordinates
logger = logging.getLogger(__name__)


class SeaSurfaceTempDownloader:
    """
    downloading monthly sst data from noaa database
    """
    def __init__(self,
                 locations: pd.DataFrame,
                 start_date: str,
                 end_date: str,
                 lat_col: str,
                 lon_col: str,
                 n_neighbors=5):
        """
        Download the sea surface temperature (SST) data from NOAA.
        If coordinates are provided: one of two things will happen:
        (1) if the coordinates are on the land, we will find the nearest the sea points
        and report the weighted average of them.
        (2) if the coordinates are in the sea, we will report the value of SST.
        """
        # mask is the layer that tells land from sea
        sst_data_url = 'ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc'
        sst_mask_url = 'ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2/lsmask.nc'

        self.temp_dir = tempfile.mkdtemp()
        sst_data_dir = os.path.join(self.temp_dir, 'sst_data.nc')
        sst_mask_dir = os.path.join(self.temp_dir, 'sst_mask.nc')

        self._locations = locations
        self.start_year, self.start_month, self.start_day = self._input_parser(start_date)
        self.end_year, self.end_month, self.end_day = self._input_parser(end_date)
        self.start_date = date(*self._input_parser(start_date, type_required='int'))
        self.end_date = date(*self._input_parser(end_date, type_required='int'))

        self._locations = locations
        self.dataset = self.prepare_dataset(data_in=sst_data_url,
                                            data_out=sst_data_dir,
                                            mask_in=sst_mask_url,
                                            mask_out=sst_mask_dir)
        self.n_neighbors = n_neighbors
        self.lat_col = lat_col
        self.lon_col = lon_col

    @property
    def locations(self):
        if not isinstance(self._locations, pd.DataFrame):
            raise TypeError('only data frame objects are accepted')

        if self.lat_col not in self._locations:
            raise KeyError(f'{self.lat_col} is not in the input data frame')

        if self.lon_col not in self._locations:
            raise KeyError(f'{self.lon_col} is not in the input dataframe')

        return self._locations

    @classmethod
    def from_tuples(cls,
                    locations: Union[tuple, list],
                    start_date: str,
                    end_date: str,
                    n_neighbors=5):

        location_peek = locations[0]
        if isinstance(location_peek, tuple) or isinstance(location_peek, list):
            locations_df = pd.DataFrame(locations)
            logger.critical('we assume the first arg is latitude and second arg is longitude')
        else:
            if isinstance(location_peek, float) or isinstance(location_peek, int):
                locations_df = pd.DataFrame([locations])
            else:
                raise TypeError('only two types of input are supported: 1. [10, 20] or [[10, 20], [30, 30]]')

        locations_df.columns = ['LAT', 'LON']
        return cls(locations=locations_df,
                   start_date=start_date,
                   end_date=end_date,
                   lat_col='LAT',
                   lon_col='LON',
                   n_neighbors=n_neighbors)

    @staticmethod
    def _input_parser(string_date, type_required='int'):
        """
        parse year, month and day from the input
        """
        if '-' in string_date:
            string_date_ = string_date.split('-')
            if len(string_date_) == 3:
                year, month, day = string_date_
            elif len(string_date_) == 2:
                year, month = string_date_
                day = '01'
            else:
                raise ValueError('only accept string formats like 1990-09 or 1990-09-09')
            if type_required == 'int':
                return int(year), int(month), int(day)
            else:
                return year, month, day
        else:
            raise ValueError('no dash in date format. '
                             'However, we only accept formats like 1990-09 or 1990-09-09')

    @staticmethod
    def prepare_dataset(data_in, data_out, mask_in, mask_out):
        def _trigger_download(url_in, local_dir_out):
            logger.critical(f'start downloading from {url_in}')
            with closing(request.urlopen(url=url_in)) as reader:
                with open(local_dir_out, 'wb') as writer:
                    shutil.copyfileobj(reader, writer)
            logger.critical(f'downloading is done. And the file has been saved to {local_dir_out}')

        if os.path.exists(data_out):
            logger.critical(f'the file {data_out} already exists. Skip downloading')
        else:
            _trigger_download(data_in, data_out)

        if os.path.exists(mask_out):
            logger.critical(f'the file {mask_out} already exists. Skip downloading')
        else:
            _trigger_download(mask_in, mask_out)

        sst_data = Dataset(data_out, mode='r')
        sst_mask = Dataset(mask_out, mode='r')
        sst_mask = np.array(sst_mask['mask'][0].data, dtype=np.bool)

        lat = sst_data.variables['lat'][:]
        lon = sst_data.variables['lon'][:]
        sst = sst_data.variables['sst'][:]
        time = sst_data.variables['time'][:]
        lon = np.array([term if term <= 180 else (term - 360) for term in lon])

        return {'dataset': sst_data, 'mask': sst_mask, 'lat': lat,
                'lon': lon, 'time': time, 'sst': sst}

    @property
    def time_index(self):
        """
        the raw data stores time in "seconds after year 1800". Hence, we translate the user-friendly input
        to this format
        """
        timeline = self.dataset['time']
        converted_dates = {date(1800, 1, 1) + timedelta(int(t)): i for i, t in enumerate(timeline)}

        all_dates_in_between = pd.date_range(start=self.start_date, end=self.end_date)
        date_dict = {d: converted_dates.get(date(d.year, d.month, d.day), None) for d in all_dates_in_between}
        date_dict_ = {k: v for k, v in date_dict.items() if v is not None}

        if date_dict_:
            return date_dict_
        else:
            raise ValueError('Failed to find the time index corresponding '
                             'to the time you provided. '
                             'The database has only monthly data from 1981')

    def download_one_month(self, time_index):
        # noland！should be around 70% percent of original data
        base = self.locations.copy()

        sst_df_sea_only = pd.DataFrame(self.dataset['sst'].data[time_index, :, :])
        sst_df_sea_only.index = self.dataset['lat']
        sst_df_sea_only.columns = self.dataset['lon']

        sst_df_sea_only_ = sst_df_sea_only.stack().reset_index()
        sst_df_sea_only_.columns = ['LAT', 'LON', 'SST']

        cartesian_coord_trees = convert_coordinates(sst_df_sea_only_)
        tree = KDTree(cartesian_coord_trees.values)
        cartesian_coord_query = convert_coordinates(self.locations)
        distances, idx = tree.query(cartesian_coord_query.values, k=self.n_neighbors)

        sst_list = []
        for i in range(len(self.locations)):
            sst_list.append(sst_df_sea_only_.iloc[idx[i]].mean().SST)
        base['SST'] = sst_list
        return base

    def download(self):
        snapshots = []
        for date_name, index_on_timeline in self.time_index.items():
            self.dataset['sst'].data[index_on_timeline, :, :][~self.dataset['mask']] = np.nan
            df_single_month = self.download_one_month(index_on_timeline)
            df_single_month['MONTH'] = date_name.month
            df_single_month['YEAR'] = date_name.year
            snapshots.append(df_single_month)

        output = pd.concat(snapshots, ignore_index=True)
        os.remove(os.path.join(self.temp_dir, 'sst_data.nc'))
        os.remove(os.path.join(self.temp_dir, 'sst_mask.nc'))
        return output


def download_sst(start_date, end_date, locations, lat='LAT', lon='LON'):
    if isinstance(locations, pd.DataFrame):
        if lat not in locations or lon not in locations:
            raise ValueError(f'the dataframe does not a LAT column or a LON columns')
        return SeaSurfaceTempDownloader(locations=locations,
                                        lat_col=lat,
                                        lon_col=lon,
                                        start_date=start_date,
                                        end_date=end_date).download()

    elif isinstance(locations, list) or isinstance(locations, tuple):
        return SeaSurfaceTempDownloader.from_tuples(locations=locations,
                                                    start_date=start_date,
                                                    end_date=end_date).download()
    else:
        raise TypeError('the input of location can only be one the three: dataframe, list, tuple')


download_sea_surface_temperature = download_sst


if __name__ == '__main__':
    import pkg_resources
    daily_flood_file = pkg_resources.resource_stream('asset.sample_data_sc', 'GAGE-20110101-20161231-SC.parquet')
    daily_flood = pd.read_parquet(daily_flood_file)
    locations_ = daily_flood[['LAT', 'LON']].drop_duplicates().reset_index(drop=True)
    test_tuple = [10, 10]
    test_1 = download_sst(locations=locations_, lat='LAT', lon='LON', start_date='2014-01', end_date='2015-01')
    test_2 = download_sst(locations=test_tuple, start_date='2014-01', end_date='2015-01')
