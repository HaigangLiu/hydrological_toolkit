import logging
import os
import tempfile
from collections import defaultdict
from shutil import rmtree
from time import sleep
from typing import List, Union

import pandas as pd
import requests
from tqdm import tqdm

from ..data.util import get_site_info

logger = logging.getLogger(__name__)


class HydrologicalDataDownloader:
    """This class helps us to parse info from the following link, for example, algorithmically
    "https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02153051&referred_module=sw&period=&begin_date" \
    "=2017-11-20&end_date=2018-11-20"

    We will first determine which site to download:
        1. if the input type is a state name, we will send a request to USGS to get all available stations
        2. if the input type is lat lon, we will extend the lat and lon input to a bounding box, and send
        the bounding box to USGS. And find the closet location, and send it back to the user
        3. if user input the site number, we will send the site number directly to usgs.

    on a high level, no matter what user wants as input (state name or station lat and lon), we have to convert to
    station id for the final query
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 station_info: dict,
                 variable_name: str = 'GAGE',
                 variable_id: str = '00065',
                 verbose: bool = False):
        """
        :param start_date: first date, inclusive
        :param end_date: last date, inclusive
        :param station_info: key is the station number and the value is the downloadable url
        :param verbose: print more info about the downloading process
        """
        self.SCHEMA = {'SITENUMBER': str,  # make sure it is string since sometimes site number starts with 0
                       'STATION_NAME': str,
                       'LAT': float,
                       'LON': float,
                       'DATE': str,
                       f'{variable_name}_MAX': float,
                       f'{variable_name}_MIN': float,
                       f'{variable_name}_MEAN': float,
                       f'{variable_name}_SUM': float,
                       }

        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self.station_list = station_info
        self.temp_dir = tempfile.mkdtemp()
        self.variable_id = variable_id
        self.variable_name = variable_name

    @classmethod
    def from_list_of_sitenumber(cls,
                                start_date: str,
                                end_date: str,
                                station_list: List[str],
                                variable_name='GAGE',
                                variable_id='00065',
                                verbose=False):
        info = get_site_info(station_list)  # because station info asks not only site number but also lat and lon
        return cls(start_date=start_date,
                   end_date=end_date,
                   station_info=info,
                   variable_name=variable_name,
                   variable_id=variable_id,
                   verbose=verbose)

    @classmethod
    def from_state_name(cls,
                        start_date: str,
                        end_date: str,
                        state_name: str,
                        variable_name='GAGE',
                        variable_id='00065',
                        verbose: bool = False):
        """
        this will be revoked if user supplies a state name such as sc
        """
        query = f'https://waterdata.usgs.gov/nwis/nwismap?state_cd={state_name}&format=sitefile_output' \
                f'&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm' \
                f'&column_name=dec_lat_va&column_name=dec_long_va'
        list_of_stations = HydrologicalDataDownloader.query_parser(query)
        return cls(start_date=start_date,
                   end_date=end_date,
                   station_info=list_of_stations,
                   variable_id=variable_id,
                   variable_name=variable_name,
                   verbose=verbose)

    @classmethod
    def from_lat_lon_box(cls,
                         start_date: str,
                         end_date: str,
                         lat_min: Union[float, int],
                         lat_max: Union[float, int],
                         lon_min: Union[float, int],
                         lon_max: Union[float, int],
                         variable_name='GAGE',
                         variable_id='00065',
                         verbose: bool = False):
        """
        alternative constructor based on the lat lon box
        """
        query = f'https://waterdata.usgs.gov/nwis/nwismap?nw_longitude_va={lon_min}&nw_latitude_va={lat_min}&' \
                f'se_longitude_va={lon_max}&se_latitude_va={lat_max}&coordinate_format=dms&group_key=NONE&format=' \
                f'sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=' \
                f'station_nm&list_of_search_criteria=lat_long_bounding_box&column_name=dec_lat_va' \
                f'&column_name=dec_long_va'
        list_of_stations = HydrologicalDataDownloader.query_parser(query)
        logger.critical(f'there are {len(list_of_stations)} locations in the lat-lon box you specified')
        if verbose:
            logger.critical(f'here is the info of the stations: (set verbose=False to turn this off)')
            for station_name in list_of_stations.values():
                logger.critical(f'{station_name}')
        return cls(start_date=start_date,
                   end_date=end_date,
                   station_info=list_of_stations,
                   variable_id=variable_id,
                   variable_name=variable_name,
                   verbose=verbose)

    @classmethod
    def from_dataframe(cls,
                       start_date: str,
                       end_date: str,
                       df: pd.DataFrame,
                       lat: str = 'LAT',
                       lon: str = 'LON',
                       variable_name='GAGE',
                       variable_id='00065',
                       verbose: bool = False):

        lat_min = df[lat].min()
        lat_max = df[lat].max()
        lon_min = df[lon].min()
        lon_max = df[lon].max()
        return cls.from_lat_lon_box(start_date=start_date,
                                    end_date=end_date,
                                    lat_min=lat_min,
                                    lat_max=lat_max,
                                    lon_min=lon_min,
                                    lon_max=lon_max,
                                    variable_name=variable_name,
                                    variable_id=variable_id,
                                    verbose=verbose)

    @staticmethod
    def query_parser(query: str, eight_digit_station_only: bool = True):
        """
        parse the link and find downloadable stations
        the query is a full url where meta data for stations are stored
        """
        dict_of_stations = {}
        for line in requests.get(query).text.split('\n'):
            if line.startswith('USGS'):
                site_number, name, latitude, longitude = line.split('\t')[1:5]
                if eight_digit_station_only:
                    if len(site_number) == 8:
                        dict_of_stations[site_number] = (name, latitude, longitude)
                else:
                    dict_of_stations[site_number] = (name, latitude, longitude)
        return dict_of_stations

    @property
    def ref_table(self):
        ref_table = pd.DataFrame(self.station_list).T
        ref_table.columns = ['STATION_NAME', 'LAT', 'LON']
        return ref_table

    def page_parser(self, site_number: str):
        sleep(0.5)
        """
        parse the url and save the result into a default dict
        allow users to add additional information like latitude and longitude
        the additional information has to be a dict

        Note that if the file has 2 lines, the additional info will be
        appended to each of them.
        """
        template = f"https://waterdata.usgs.gov/nwis/dv?cb_{self.variable_id}=on&format=rdb&site_no={site_number}" \
                   f"&referred_module=sw&period=&begin_date={self.start_date}&end_date={self.end_date} "
        try:
            content = requests.get(template).text
        except requests.exceptions.SSLError or requests.exceptions.ConnectionError or ConnectionRefusedError:
            if self.verbose:
                logger.critical('encountered a bad connection, will pass this one')
            return None

        if content:
            content = content.split('\n')
        else:
            logger.critical(f'Got an empty page; skip this one')
            return None

        # initialize everything
        counter_num_days, header_not_set, keys, output_dict = 0, True, None, defaultdict(list)

        while content:
            line = content.pop(0)
            if header_not_set and line.startswith('agency_cd'):
                keys = line.split('\t')
                header_not_set = False  # should be only 1 header column

            elif line.startswith('USGS'):
                if keys:
                    values = line.split('\t')
                    for key, value in zip(keys, values):
                        output_dict[key].append(value)
                    counter_num_days += 1
                else:
                    raise ValueError('content comes before header. Should not happen')
            else:
                pass

        if output_dict:
            dir_ = os.path.join(os.getcwd(), self.temp_dir)
            file_ = os.path.join(dir_, site_number + '.csv')
            pd.DataFrame(output_dict).to_csv(file_)
        else:
            if self.verbose:
                logger.critical(f'no valid data found for station id {site_number}')

    def _finish_up_and_tear_down(self):
        header = {'site_no': 'SITENUMBER', 'datetime': 'DATE'}
        code_to_summary_type = {'00001': 'MAX',
                                '00002': 'MIN',
                                '00003': 'MEAN',
                                '00006': 'SUM'}
        container = []
        for filename in os.listdir(self.temp_dir):
            if filename.endswith('.csv'):
                file_abs_dir = os.path.join(self.temp_dir, filename)
                df = pd.read_csv(file_abs_dir, index_col=0, dtype={'site_no': str})
                os.remove(file_abs_dir)
                for column in df.columns:
                    for code in code_to_summary_type:
                        if column.endswith(f'{self.variable_id}_{code}'):
                            header[column] = f'{self.variable_name}_{code_to_summary_type[code]}'
                flag_cols = [col for col in df.columns if col.endswith('cd')]
                df.rename(header, axis=1, inplace=True)
                df.drop(columns=flag_cols, inplace=True)

                _, columns_count = df.shape
                if 3 <= columns_count <= 5:
                    # less than 3 columns means there is no data just location and date info
                    # more than 5 means there are multiple versions of the variable; disregard this kind
                    container.append(df)
        rmtree(self.temp_dir)
        try:
            return pd.concat(container)
        except ValueError as e:
            if 'No objects to concatenate' in str(e):
                return None
            else:
                raise

    @staticmethod
    def populate_dates_with_na(df: pd.DataFrame,
                               date_col: str,
                               id_col: str):
        """
        fill the missing dates with na. Because some of the dates are not aligned. For instance, some places might
        only got Feb and other places only got Jan. We use the superset and populate everywhere else with null values
        """
        logger.critical(f'the size of the dataset before populating null value is {len(df)}')

        df[date_col] = pd.to_datetime(df[date_col])
        complete_dates = pd.date_range(df[date_col].min(), df[date_col].max())
        index_reconstructed = pd.MultiIndex.from_product([df[id_col].unique().tolist(), complete_dates])
        df.set_index([id_col, date_col], inplace=True)
        logger.critical(f'the size of the dataset after aligning the dates is {len(df)}')

        df = df.reindex(index_reconstructed).reset_index()
        df.rename({'level_0': 'SITENUMBER', 'level_1': 'DATE'}, axis=1, inplace=True)
        return df

    def download(self, single_thread: bool = False):
        """
        a major issue here is: if we hit the site too fast too many times. It will refuse to connect.
        the way to get around this
        (1) use the single threaded version so not so many requests are sent or
        (2) use multiple processing but make sure sleep a bit; also use async.
        """
        jobs_site_number = list(self.station_list.keys())
        from multiprocessing import Pool, cpu_count

        if not single_thread:  # async
            with Pool(cpu_count()) as process_executor:
                result = process_executor.map_async(self.page_parser, jobs_site_number)
                while not result.ready():
                    if self.verbose:
                        logger.critical("Remaining locations to be processed: {}".format(result._number_left))
                    result.wait(timeout=0.5)
        else:
            for job in tqdm(jobs_site_number):
                self.page_parser(job)

        final_df = self._finish_up_and_tear_down()

        if final_df is None:
            import warnings
            warnings.warn('got an empty dataframe. either the date range is not available, '
                          'or such variable type does not exist in the USGS database')
            return final_df

        else:
            final_df = self.populate_dates_with_na(df=final_df,
                                                   date_col='DATE',
                                                   id_col='SITENUMBER')

            # note that some kinds of variables like prcp only has sum
            # but gage only has min, max and mean, which mean most variables do
            # not have exactly everything
            measure_cols = [col for col in self.SCHEMA if col.startswith(self.variable_name) and col in final_df]
            other_info = ['SITENUMBER', 'STATION_NAME', 'LAT', 'LON', 'DATE']
            final_df = pd.merge(left=final_df,
                                right=self.ref_table,
                                left_on=['SITENUMBER'],
                                right_index=True,
                                how='left')[other_info + measure_cols]. \
                reset_index()

            for potential_col in self.SCHEMA.copy().keys():
                if potential_col not in measure_cols and potential_col not in other_info:
                    # get rid of the column before apply the schema
                    del self.SCHEMA[potential_col]
            final_df = final_df.astype(self.SCHEMA)

            # if, for example, gage_mean, gage_max and gage_min are all missing
            # we will delete this column
            final_df.dropna(subset=measure_cols, how='all', inplace=True)
            return final_df


def download_hydrological_variable(start_date: str,
                                   end_date: str,
                                   variable_name='GAGE',
                                   variable_id='00065',
                                   locations=None,
                                   **kwargs):
    """
    Here's a list of supported input as locations:
    1. a pair of coordinate [34, -82],
    2. a list of coordinates,
    3. a list of site numbers make sure they are string type,
    4. a string representing an acronym of a state such as SC,
    5. a string that is an address,
    6. a pandas dataframe with columns LAT and LON
    """
    if isinstance(locations, str) and len(locations) == 2:
        return HydrologicalDataDownloader. \
            from_state_name(start_date=start_date,
                            variable_name=variable_name,
                            variable_id=variable_id,
                            end_date=end_date,
                            state_name=locations).download()

    elif isinstance(locations, list) and isinstance(locations[0], str):
        return HydrologicalDataDownloader. \
            from_list_of_sitenumber(start_date=start_date, end_date=end_date, station_list=locations).download()

    elif locations is None:
        if 'lat_min' not in kwargs or 'lat_max' not in kwargs or 'lon_min' not in kwargs or 'lon_max' not in kwargs:
            raise KeyError(f'require lat_min, lat_max, lon_min and lon_max')
        lat_min, lat_max, lon_min, lon_max = kwargs['lat_min'], kwargs['lat_max'], kwargs['lon_min'], kwargs['lon_max']
        return HydrologicalDataDownloader.from_lat_lon_box(start_date=start_date,
                                                           end_date=end_date,
                                                           variable_name=variable_name,
                                                           variable_id=variable_id,
                                                           lat_max=lat_max,
                                                           lon_max=lon_max,
                                                           lat_min=lat_min,
                                                           lon_min=lon_min).download()

    elif isinstance(locations, list) and isinstance(locations[0], list) and len(locations) == 1:
        logger.critical('we treat the input as [lat, lon] coordinates since the list contains float number or int ')
        logger.critical("if you wish to download by site number, make sure pass a list of strings like ['02110400']")
        # pair of coord
        locations = locations[0]
        lat_max, lat_min = locations[0] + 1, locations[0] - 1
        lon_max, lon_min = locations[1] + 1, locations[1] - 1
        logger.critical(f'we have expanded your input to a bounding box from {lat_min} to {lat_max} in lat, '
                        f'and {lon_min} to {lon_max} in lon')

        return HydrologicalDataDownloader.from_lat_lon_box(start_date=start_date,
                                                           end_date=end_date,
                                                           variable_name=variable_name,
                                                           variable_id=variable_id,
                                                           lat_max=lat_max,
                                                           lon_max=lon_max,
                                                           lat_min=lat_min,
                                                           lon_min=lon_min).download()

    elif isinstance(locations, list) and isinstance(locations[0], list):  # list of coordinates
        import numpy as np
        locations_array = np.array(locations)
        lat_max, lon_max = locations_array.max(axis=0)
        lat_min, lon_min = locations_array.min(axis=0)
        return HydrologicalDataDownloader.from_lat_lon_box(start_date=start_date,
                                                           end_date=end_date,
                                                           variable_name=variable_name,
                                                           variable_id=variable_id,
                                                           lat_max=lat_max,
                                                           lon_max=lon_max,
                                                           lat_min=lat_min,
                                                           lon_min=lon_min).download()
    else:
        raise TypeError('we only support the following kind of input: '
                        '1. a pair of coordinate [34, -82] \n,'
                        '2. a list of coordinates, \n'
                        '3. a list of site numbers make sure they are string type,\n'
                        '4. a string representing an acronym of a state such as SC, \n'
                        '5. a string that is an address, \n'
                        '6. a pandas dataframe with columns LAT and LON')


from functools import partial
download_flood = partial(download_hydrological_variable,
                         variable_name='GAGE',
                         variable_id='00065')
download_streamflow = partial(download_hydrological_variable,
                              variable_name='StreamFlow',
                              variable_id='00060')
download_gauge = download_flood
download_discharge = download_streamflow
