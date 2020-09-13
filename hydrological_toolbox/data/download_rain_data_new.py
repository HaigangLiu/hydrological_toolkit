from collections import defaultdict
import pickle
import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Iterable

import tempfile
import requests
from netCDF4 import Dataset
from pandas import DataFrame, read_csv, concat
from shapely.geometry import Point
from data.util import get_state_contours, get_in_between_dates, project, get_x_y_projections

logger = logging.getLogger(__name__)


class RainDataDownLoaderAfter2017:

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 locations: dict):
        """
        This class downloads the rainfall data after 2017 since NWIS uses a different data source and format since

        Post 2017 data have a different format than the ones before.
        :param start_date: e.g., '2018-01-01'
        :param end_date: e.g., '2018-01-10'
        :param locations: e.g., 'SC', the acronym of a US state
        """

        self.state_name = locations
        self.start_date = start_date
        self.end_date = end_date

        self.local_dir = tempfile.mkdtemp()
        self.list_of_dates = list(get_in_between_dates(start_date=start_date, end_date=end_date))
        self.download_links = list(self._generate_download_links())
        self.locations = locations

    def _generate_download_links(self) -> Iterable:
        """
        generate the pair of remote file and local file
        """
        for date_ in self.list_of_dates:
            year, month, day = date_.split('-')
            all_ = ''.join([year, month, day])
            template = f'https://water.weather.gov/precip/downloads/{year}/{month}/{day}/nws_precip_1day_{all_}_conus.nc'
            yield template, os.path.join(self.local_dir, all_ + '.nc')  # the latter one is the local filename

    @classmethod
    def from_locations(cls, start_date, end_date, locations):
        """
        this alternative constructor allows user to provide a list of lat long or a data frame
        """
        if isinstance(locations, list) or isinstance(locations, DataFrame):
            x_y_lat_lon = get_x_y_projections(locations)
            locations_lookup = {}
            for _, row in x_y_lat_lon.iterrows():
                locations_lookup[(int(row['x']), int(row['y']))] = (row['lat'], row['lon'])
            # e.g. {(12, 13): (lat, lon)}
            return cls(start_date=start_date, end_date=end_date, locations=locations_lookup)

    @classmethod
    def from_state_name(cls, start_date, end_date, state_name):
        example_file = '../asset/new_data/nws_precip_1day_20190401_conus.nc'
        example_file = Dataset(example_file)
        x = example_file['x'][:]
        y = example_file['y'][:]
        baseline_coord = example_file['crs'].proj4  # projection info
        state_contour = get_state_contours(state_name, lat_first=True)
        # self.locations = self._find_valid_locations_in_this_state()
        """
        this location list is a list of tuple. Each tuple is an index.
        And the value corresponds to this index resides inside the state
        """
        dir_to_location_cache = f'NCEP Stage IV Daily {state_name}.pkl'
        if os.path.exists(dir_to_location_cache):
            logger.critical('loading valid locations from cache. This should be relatively fast.')
            with open(dir_to_location_cache, 'rb') as handle:
                cached_locations = pickle.load(handle)
                return cls(locations=cached_locations, start_date=start_date, end_date=end_date)

        logger.critical(f'cached locations for {state_name} do not exist; rebuilding the cache')
        cached_locations = {}
        state_contour_converted = project(state_contour,
                                          original_prj='epsg:4326',
                                          new_prj=baseline_coord)

        # generate a meshgrid (cartesian) for x and y
        xx = list(range(1121)) * 881
        yy_nested = ([i] * 1121 for i in range(881))
        yy = (item for sublist in yy_nested for item in sublist)

        geo_points = []
        for pair in zip(xx, yy):
            x_idx, y_idx = pair
            # convert to Point object to use .within method
            geo_point = Point(x[x_idx], y[y_idx])

            if geo_point.within(state_contour_converted):
                cached_locations[tuple([x_idx, y_idx])] = geo_point
                geo_points.append(geo_point)

        cached_locations_ = {}
        geo_points_in_lat_lon = project(shapely_object_or_list=geo_points,
                                        original_prj=baseline_coord,
                                        new_prj='epsg:4326')

        for key, value in zip(cached_locations, geo_points_in_lat_lon):
            cached_locations_[key] = value

        with open(dir_to_location_cache, 'wb') as handle:
            pickle.dump(cached_locations_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.critical(f'finished building the cache. Index of cached locations saved to {dir_to_location_cache}')
        return cls(locations=cached_locations_, start_date=start_date, end_date=end_date)

    @staticmethod
    def file_is_valid(netCDF_file: Dataset) -> bool:
        """
        we assume the netCDF data has a column called observation and
        the observation is 881 x 1121. We need to validate this assumption nevertheless.
        """
        observations = netCDF_file.variables['observation']
        dim_a, dim_b = observations.shape
        if dim_a != 881:
            raise ValueError('the first dimension should be of size 881')
        if dim_b != 1121:
            raise ValueError('the second dimension should be of size 1121')
        return True

    def download_singular_file(self, remote_dir: str, local_dir: str) -> None:
        """
        download a singular file, filter down and only keep the current state (such as SC)
        and save it to a csv file
        :param remote_dir: the link to grab data
        :param local_dir: the local .nc file to store data
        """
        downloaded_content = requests.get(remote_dir).content
        with open(local_dir, 'wb') as f:
            f.write(downloaded_content)

        netCDF_file = Dataset(local_dir)

        if self.file_is_valid(netCDF_file):
            rainfall_location_and_values = defaultdict(list)
            observations = netCDF_file.variables['observation'][:]

            for dim_x_idx, dim_y_idx in self.locations.keys():
                # the observation is in (y, x) format
                rainfall_location_and_values['x'].append(dim_x_idx)
                rainfall_location_and_values['y'].append(dim_y_idx)
                try:
                    rainfall_location_and_values['prcp'].append(round(observations[dim_y_idx][dim_x_idx], 4))
                except TypeError as e:
                    if str(e) == "type MaskedConstant doesn't define __round__ method":
                        logger.critical('Probably this location is not in the United States')
                        raise
                geometry = self.locations[tuple([dim_x_idx, dim_y_idx])]
                rainfall_location_and_values['lat'].append(geometry[0])
                rainfall_location_and_values['lon'].append(geometry[1])

            local_csv = local_dir.replace('.nc', '.csv')
            df_one_file = DataFrame(rainfall_location_and_values)
            *_, date_ = os.path.split(local_dir)
            df_one_file['date'] = date_.replace('.nc', '')  # the date column happens to be the filename
            df_one_file.to_csv(local_csv)

    def download(self, single_thread=True):
        if not single_thread:
            with Pool(cpu_count()) as process_executor:
                result = process_executor.starmap_async(self.download_singular_file, list(self.download_links))
                while not result.ready():
                    logger.critical("num left: {}".format(result._number_left))
                    result.wait(timeout=0.5)
        else:
            for remote_dir, local_dir in self._generate_download_links():
                self.download_singular_file(remote_dir=remote_dir, local_dir=local_dir)

        df_list = []
        for remote_and_local in self.download_links:
            # [0] is remote and [1] is local
            csv_dir = remote_and_local[1].replace('.nc', '.csv')
            df_list.append(read_csv(csv_dir, index_col=0))
            os.remove(csv_dir)
        return concat(df_list)


def caller(start_date, end_date, locations, **kwargs):
    if isinstance(locations, str) and len(locations) == 2:
        # state abbreviation
        return RainDataDownLoaderAfter2017.\
            from_state_name(state_name=locations, start_date=start_date, end_date=end_date).download()

    elif isinstance(locations, str):
        from data.util import get_lat_lon_from_string
        lat, lon = get_lat_lon_from_string(locations)
        logger.critical(f'this is the latitude and longitude info: {lat} and {lon}')
        logger.critical(f'verify this is accurate')
        return RainDataDownLoaderAfter2017.\
            from_locations(start_date=start_date, end_date=end_date, locations=[[lat, lon]]).download(single_thread=True)

    elif isinstance(locations, list):
        return RainDataDownLoaderAfter2017. \
            from_locations(start_date=start_date, end_date=end_date, locations=locations).download(single_thread=True)

    elif isinstance(locations, DataFrame):
        if 'LAT' not in locations or 'LON' not in locations:
            if 'lat' in kwargs and 'lon' in kwargs:
                lat, lon = kwargs['lat'], kwargs['lon']
            else:
                raise KeyError('the dataframe must have a LAT and a LON column; '
                               'or you can specify your own lat lon column with lat and lon args')
        else:
            lat, lon = 'LAT', 'LON'
        locations = locations[[lat, lon]]  # just to make sure lat comes before lon
        locations = list(locations.values)
        return RainDataDownLoaderAfter2017. \
            from_locations(start_date=start_date, end_date=end_date, locations=locations).download(single_thread=True)
    else:
        raise TypeError('unsupported type!')

download_rain_after_2017 = caller


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    # downloader = RainDataDownLoaderAfter2017.from_locations('2017-01-01', '2017-01-02', locations=[[24, -83]]).download()
    downloader = download_rain_after_2017('2017-01-01', '2017-01-02', locations='Columbia, SC')
    print(downloader)

    # print(downloader)
    # 461.76522265899996
    end = perf_counter()
    print(end - start)
