from functools import cached_property
import logging
import os
import re
import requests
import tarfile
import shutil
import tempfile
from typing import List, Union, Tuple
import warnings

import geopandas as gpd
import pandas as pd
from scipy.spatial import KDTree

from ..data.util import get_in_between_dates, get_lat_lon_from_string

logger = logging.getLogger(__name__)


class RainDataDownloader:
    SCHEMA = {'LAT': float,
              'LON': float,
              'DATE': str,
              'PRCP': float}

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 max_tries: int = 10,
                 verbose: bool = True,
                 locations: Union[List[List[float]], pd.DataFrame] = None):
        """
        user can provide either a list of (lat, lon) coordinates or a shorthand name of a state

        if state name: we will find all the available nws stations inside that state
        if locations: we will find the nearest available location for each of the location.

        :param start_date: first day of data download (inclusive)
        :param end_date: last day of data download (inclusive)
        :param max_tries: max tries until time out
        """
        self.var_name = 'globvalue'  # the default column name
        self.web_loc = 'https://water.weather.gov/precip/archive'
        self.temp_dir = tempfile.mkdtemp()

        self._customized_location_list = locations
        self.start_date, self.end_date = start_date, end_date
        self.max_tries = max_tries
        self.dates_list = get_in_between_dates(self.start_date, self.end_date)
        self.links_download_and_store = list(self._generate_links_to_download())
        self.verbose = verbose

        if not self.links_download_and_store:
            raise ValueError('the jobs list for downloading is empty. '
                             'Double check the date. For instance, if the starting date is '
                             'later than ending date, this error would show up.')

    @staticmethod
    def load_all_rainfall_locations():
        """
        get a complete list of nws locations (grid) since only these locations actually have data on them.
        the way we use it is to find the closest nws location to the location provided by the user.
        """
        from ..data.util import nws_locations_nation_wide
        locs = pd.read_parquet(nws_locations_nation_wide)
        locs.rename({'lat': 'LAT', 'lon': 'LON', 'x': 'HRAPX', 'y': 'HRAPY'}, axis=1, inplace=True)
        return locs

    @classmethod
    def from_state_name(cls, state_name, start_date: str, end_date: str, max_tries: int = 10, verbose: bool = True):
        """
        The location list can be represented as hrapx, and hrapy (that's all we need), which can be used to
        uniquely represent a point in the NWS grid. This would avoid geometry-based merging.
        Performance is much better this way.

        """
        from ..data.util import state_contours, get_state_name_from_abbreviation
        state_contours = gpd.read_file(state_contours)
        state_name = get_state_name_from_abbreviation(state_name)
        state_contours = state_contours[state_contours.NAME == state_name]
        df = RainDataDownloader.load_all_rainfall_locations()
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT))

        # find the locations (1) belong to nws and (2) belong to the state
        statewide_rainfall_data = gpd.sjoin(left_df=gdf,
                                            right_df=state_contours,
                                            op='within',
                                            how='inner') \
            .drop(columns=['index_right'])
        logger.info(f'location information loading is done')
        statewide_rainfall_data = statewide_rainfall_data[['HRAPX', 'HRAPY', 'LAT', 'LON']].reset_index(drop=True)

        return cls(start_date=start_date,
                   end_date=end_date,
                   max_tries=max_tries,
                   verbose=verbose,
                   locations=statewide_rainfall_data)

    @cached_property
    def locations_to_download(self) -> pd.DataFrame:
        """
        get the nws locations that needs downloading.
        """
        if isinstance(self._customized_location_list, pd.DataFrame):
            return self._customized_location_list
        else:
            return self._find_nearest_points_on_nws_map(self._customized_location_list)

    @staticmethod
    def _find_nearest_points_on_nws_map(locations: Union[List[List[float]], List[Tuple[float]]]):
        """
        based on the user input determine the closest available location to download
        this method will be invoked only when user passed a list of locs instead of state name
        """
        if not isinstance(locations, list):
            raise TypeError('locations are supposed to a list of coordinates')

        logger.critical('looking for the closest locations to the location you provide')
        logger.critical('make sure the latitude is coming before longitude.')

        all_rainfall_locations = RainDataDownloader.load_all_rainfall_locations()
        _, indices = KDTree(all_rainfall_locations[['LAT', 'LON']].values).query(locations)
        locations_in_the_us = all_rainfall_locations.iloc[indices]
        locations_in_the_us = locations_in_the_us[['HRAPX', 'HRAPY', 'LAT', 'LON']]
        return locations_in_the_us

    def _generate_links_to_download(self):
        """
        generate the dir for input (web url from the nws archive) and the output (local download dir)
        the result format is a list of tuple for all the dates in between

        Note that the local dir is absolute dir.
        """
        for date_ in self.dates_list:
            year, month, day = re.findall(r'\d+', date_)
            date_specific_part = f"{year}/{month}/{day}/nws_precip_1day_observed_shape_{date_.replace('-', '')}.tar.gz"
            file_name_tar = ''.join([date_.replace('-', ''), '.tar.gz'])  # like 19900909.tar.gz

            web_url = os.path.join(self.web_loc, date_specific_part)
            file_name_tar = os.path.join(self.temp_dir, file_name_tar)
            local_dir = os.path.join(self.temp_dir, file_name_tar)
            yield web_url, local_dir

    @staticmethod
    def download_and_extract(web_url: str, local_dir: str):
        """
        Download from the web url generated before.
        Since the raw file is a tar file, and we will extract it.
        After processing, this method will return a status boolean, and directory of extracted files.
        """
        request_sent = requests.get(web_url)
        if not request_sent.ok:
            warnings.warn(f'the status code is {request_sent.status_code}')
            if request_sent.status_code == 404:
                warnings.warn(f'the data you request might not be available. Possible reason: too old or too new.')
            return False, ''
        try:
            downloaded_content = request_sent.content
            with open(local_dir, 'wb') as f:
                f.write(downloaded_content)
                dir_, fname = os.path.split(local_dir)
                target_folder_name, *_ = re.findall(r'\d{8}', fname)  # 8-number, should be date
            with tarfile.open(local_dir) as tar:
                tar.extractall(os.path.join(dir_, target_folder_name))

        except EOFError or tarfile.ReadError:
            pass

        finally:
            os.remove(local_dir)  # tear down
            abs_target_folder = os.path.join(dir_, target_folder_name)

            file_number = 0
            local_dir_remove_tar_gz = local_dir.replace('.tar.gz', '')
            for _ in os.listdir(local_dir_remove_tar_gz):
                file_number = file_number + 1

            if file_number != 4:  # not complete if <4 files in the folder and gpd will not be able to read it.
                return False, ''
            else:
                return True, abs_target_folder  # the local folder after extraction, the format is like '20090101'

    def convert_downloaded_binary_to_csv(self, downloaded_rainfall_folder: str) -> None:
        try:
            rain_data_daily = gpd.read_file(downloaded_rainfall_folder)
        except ValueError:
            logger.critical(f'content in {downloaded_rainfall_folder} is not complete. Skip this job')
            return None
        finally:
            shutil.rmtree(downloaded_rainfall_folder)

        # there are variations such as Hrapx and HRAPX. we force the lower case letter across board
        rain_data_daily.columns = [c.lower() for c in rain_data_daily.columns]

        for column in ['hrapx', 'hrapy', 'globvalue']:
            if column not in rain_data_daily:
                logger.critical(f'{column} is not in the data base. This one is skipped')
                return None
            else:
                rain_data_daily = rain_data_daily[['hrapx', 'hrapy', 'globvalue']]

        statewide_rainfall_data = pd.merge(rain_data_daily,
                                           self.locations_to_download,
                                           left_on=['hrapx', 'hrapy'],
                                           right_on=['HRAPX', 'HRAPY'],
                                           how='right',
                                           suffixes=['', '_'])

        statewide_rainfall_data.drop(columns=['hrapx', 'hrapy'], inplace=True)

        null_values = statewide_rainfall_data[self.var_name].isnull().sum()
        if self.var_name in statewide_rainfall_data:
            statewide_rainfall_data.fillna({self.var_name: 0}, inplace=True)

        processed_file_name = downloaded_rainfall_folder + '.csv'
        logger.critical(f'there are {len(statewide_rainfall_data)} records | '
                        f'missing value counts: {null_values} | '
                        f'missing values are filled with 0 because no rain were measured')

        statewide_rainfall_data.to_csv(processed_file_name)

    def download_controller(self, link: str, target_dir: str) -> None:
        """
        This method tell download_and_extract() when to stop.
        either after 20 tries, or finished downloading successfully (i.e. the flag is_success = True)
        after that, it will convert the downloaded folder/binary into csv and store it in temp dir.
        """
        tries = 0
        job_date = target_dir.split('/')[-1].replace('.tar.gz', '')

        is_success = False
        folder_name = None
        while tries < self.max_tries and not is_success:
            is_success, folder_name = self.download_and_extract(link, target_dir)
            if self.verbose:
                logger.critical(
                    f'Try #: {tries + 1}: started downloading for {len(self.locations_to_download)} location(s)')
                tries = tries + 1
        if is_success:
            self.convert_downloaded_binary_to_csv(folder_name)
        else:
            logger.critical(f'exceeded max tries {self.max_tries} for date {job_date}, '
                            f'download failed for this date')

    def download(self, single_thread=False):
        """
        A caller to download all the download jobs in parallel.
        Returns a dataframe. User can call .to_csv() or .to_parquet() to save to local.
        """
        from multiprocessing import Pool, cpu_count
        if not single_thread:
            with Pool(cpu_count()) as process_executor:
                result = process_executor.starmap_async(self.download_controller, self.links_download_and_store)
                while not result.ready():
                    result.wait(timeout=0.5)
        # elif not single_thread:
        #     with Pool(cpu_count()) as process_executor:
        #         process_executor.starmap(self.download_controller, self.links_download_and_store)
        else:
            for web_url, local_dir in self.links_download_and_store:
                self.download_controller(link=web_url, target_dir=local_dir)

        files = []
        for _, local_file in self.links_download_and_store:
            local_file = local_file.replace('.tar.gz', '.csv')
            if os.path.exists(local_file):
                downloaded_rainfall_file_ = os.path.join(self.temp_dir, local_file)
                df = pd.read_csv(downloaded_rainfall_file_)[['LAT', 'LON', 'HRAPX', 'HRAPY', self.var_name]]

                # this is because individual files do not have date column. We
                # want to create such column when merging all files.
                date_string = local_file.split('/')[-1].split('.')[0]
                year, month, day = date_string[0:4], date_string[4:6], date_string[6:8]
                df['DATE'] = '-'.join([year, month, day])
                df.rename(columns={self.var_name: 'PRCP'}, inplace=True)
                files.append(df)
            else:
                logger.warning(f'{local_file} does not exist. '
                               f'This indicates the job corresponding to that date failed')

        output = pd.concat(files)
        output = output.astype(self.SCHEMA)
        return output


def download_rainfall(start_date, end_date, location, **kwargs):
    """
    handles different types of input differently.
    """
    if isinstance(location, str) and len(location) == 2:
        logger.critical(f'start downloading rainfall in state {location}')
        return RainDataDownloader.from_state_name(start_date=start_date,
                                                  end_date=end_date,
                                                  state_name=location).download()

    elif isinstance(location, str) and len(location) > 2:
        # must use single thread this case otherwise it's a violation of Nominatim Usage Policy.
        lat, lon = get_lat_lon_from_string(location)

        logger.critical('since the length location description is >2. We treat this as an address')
        logger.critical(f'LATITUDE: {lat} LONGITUDE: {lon} | Please verify this is accurate')

        return RainDataDownloader(start_date=start_date,
                                  end_date=end_date,
                                  locations=[[lat, lon]]) \
            .download(single_thread=True)

    elif isinstance(location, list):
        return RainDataDownloader(start_date=start_date, end_date=end_date, locations=location).download()

    elif isinstance(location, pd.DataFrame):

        try:
            locations_ = list(location[['LAT', 'LON']].values)
        except KeyError:
            if 'lat' in kwargs and 'lon' in kwargs:
                locations_ = list(location[[kwargs['lat'], kwargs['lon']]].values)
            else:
                raise KeyError('dataframe must have a LAT column and a LON column')
        return RainDataDownloader(start_date=start_date,
                                  end_date=end_date,
                                  locations=locations_).download()
    else:
        raise TypeError('the input can only be either string or a list of coordinates or a dataframe')

