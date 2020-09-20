from __future__ import annotations

import gzip
import logging
import os
import shutil
import struct
import tempfile
from functools import cached_property
from math import copysign, floor
from typing import Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AltitudeDownloader:
    """
    Download the altitude data from SRTM project by NASA.
    There are three versions based on resolutions and we used the most granular/detailed version (1 arc second)
    the input can be a tuple/list of (lat, lon) pair. or a batch job like data frame
    """

    def __init__(self,
                 locations: pd.DataFrame,
                 lat_col: str = None,
                 lon_col: str = None):

        self._locations = locations
        self.lat_col = lat_col
        self.lon_col = lon_col

        self.local_dir = tempfile.mkdtemp()
        self.links = self.generate_links_for_downloading()
        self.local_file_list = set()  # keep track of local file created; absolute dir

    @classmethod
    def from_tuple(cls, tuple_or_list: Union[tuple, list]) -> AltitudeDownloader:
        logger.critical('we assume the latitude comes before longitude in the tuple')
        peak = tuple_or_list[0]
        if isinstance(peak, tuple) or isinstance(peak, list):
            locations_df = pd.DataFrame(tuple_or_list)
        elif isinstance(peak, int) or isinstance(peak, float):
            locations_df = pd.DataFrame([tuple_or_list])
        else:
            example = '[34, 24] or [[34, 23], [23, -18]]'
            raise ValueError(f'this method only supports a list of coordinates'
                             f' or one coordinate, here is one example for each {example}')
        locations_df.columns = ['LAT', 'LON']
        logger.critical(f'since locations are formatted as a list, '
                        f'we will assume the first arg is latitude and the second is longitude')
        return cls(locations_df, lat_col='LAT', lon_col='LON')

    # both cases are handled the same way
    from_list = from_tuple

    @cached_property
    def locations(self):
        if isinstance(self._locations, pd.DataFrame):
            if self.lat_col is None or self.lon_col is None:
                raise KeyError('since the locations is a data frame, please specify the latitude column'
                               'and the longitude column by setting lat_col and lon_col args')
            elif self.lat_col not in self._locations or self.lon_col not in self._locations:
                raise KeyError(f'{self.lon_col} or {self.lat_col} is not in the dataframe')
            else:
                return self._locations

        elif isinstance(self._locations, list) or isinstance(self._locations, tuple):
            logger.critical(f'since locations are formatted as a list, '
                            f'we will assume the first arg is latitude and the second is longitude')
            if self._locations:
                if isinstance(self._locations[0], list) or isinstance(self._locations[0], tuple):
                    locations_df = pd.DataFrame(self._locations)
                    locations_df.columns = ['LAT', 'LON']
                    return locations_df
                elif isinstance(self._locations[0], float) or isinstance(self._locations[0], int):
                    return self._locations
                else:
                    example = '[34, 24] or [[34, 23], [23, -18]]'
                    raise ValueError(f'this method only supports a list of coordinates'
                                     f' or one coordinate, here is one example for each {example}')
        else:
            raise TypeError('this class only accept dataframe or list as input')

    @staticmethod
    def sign(x: Union[float, int]) -> float:
        """
        python does not have sign function for scalars. Use copy sign instead
        """
        return copysign(1, x)

    @staticmethod
    def make_format(lat: float, lon: float):
        """
        assuming the input is in the float format
        translate them into east/west and north/south for later use
        """
        if lat > 90 or lat < -90:
            raise ValueError('we assume the latitude is a float between -90 and 90')
        if lon > 180 or lon < -180:
            raise ValueError('we assume the longitude to be a number between -180 and 180')

        lat_string = 'N' + str(lat) if lat >= 0 else 'S' + str(abs(lat))
        lon_string = 'E' + str(lon).zfill(3) if lon >= 0 else 'W' + str(abs(lon)).zfill(3)  # has to be 080 format
        both = lat_string + lon_string
        return lat_string, lon_string, both

    def generate_links_for_downloading(self) -> list:
        """
        here's the rule: round to the smaller
        for locations such as (39.6, 29.5) we will refer to the tile indexed by
        (39, 29). and if it's (-39.6, 29.5), then the logic still follow through,
        we will look at (-40, 29). Hence, this is exactly what a floor function is
        for.
        """
        df = self.locations
        if isinstance(df, list) or isinstance(df, tuple):
            lat, lon = df
            lat, lon = floor(lat), floor(lon)
            lat, _, both = self.make_format(lat, lon)
            url = '/'.join([lat, both])
            return [f'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{url}.hgt.gz']

        else:
            # prevent pandas from using references
            df = self.locations.copy()

            df[self.lat_col] = np.floor(df[self.lat_col])
            df[self.lon_col] = np.floor(df[self.lon_col])
            df[self.lat_col] = df[self.lat_col].astype(int)
            df[self.lon_col] = df[self.lon_col].astype(int)

            lat_lon_pairs = df[[self.lat_col, self.lon_col]].drop_duplicates()

            list_of_links = []
            for _, tile in lat_lon_pairs.iterrows():
                lat, lon = tile[self.lat_col], tile[self.lon_col]
                lat, _, both = self.make_format(lat, lon)
                url = '/'.join([lat, both])
                list_of_links.append(f'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{url}.hgt.gz')
            return list_of_links

    def download_link(self, link: str) -> None:
        *_, temp_name = link.split('/')
        logger.info(f'start downloading from {link}')
        response = requests.get(link)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 2014
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        if response.status_code != 200:
            return None

        with open(os.path.join(self.local_dir, temp_name), 'wb') as writer:
            for data in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(data))
                writer.write(data)
        progress_bar.close()
        # unzip the file
        with gzip.open(os.path.join(self.local_dir, temp_name), 'rb') as f_in:
            with open(os.path.join(self.local_dir, temp_name.replace('.gz', '')), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"file has been saved to {temp_name.replace('.gz', '')} at {os.path.join(self.local_dir, temp_name)}")

    @staticmethod
    def _convert_to_minutes_and_seconds(lat_or_lon_in_decimal_points: float):
        """
        due to how the data are stored, the minute and second part of the location
        determines where we can find the height in a tile.
        """
        if isinstance(lat_or_lon_in_decimal_points, str):
            lat_or_lon_in_decimal_points = float(lat_or_lon_in_decimal_points)

        negative = lat_or_lon_in_decimal_points < 0
        lat_or_lon_in_decimal_points = abs(lat_or_lon_in_decimal_points)
        minutes, seconds = divmod(lat_or_lon_in_decimal_points * 3600, 60)
        degrees, minutes = divmod(minutes, 60)
        if negative:
            if degrees > 0:
                degrees = -degrees
            elif minutes > 0:
                minutes = -minutes
            else:
                seconds = -seconds
        return degrees, minutes, seconds

    def look_up_altitude(self,
                         lat: float,
                         lon: float,
                         num_rows: int = 3601,
                         arc_sec: int = 1) -> Union[float, None]:
        """
        the tricky part is the look up logic of east half and west half are slightly different
        so is north and south part.
        """
        lat_d, lon_d = floor(lat), floor(lon)

        _, lat_m, lat_s = self._convert_to_minutes_and_seconds(lat)
        _, lon_m, lon_s = self._convert_to_minutes_and_seconds(lon)

        pinpoint_lat = lat_m * 60 + lat_s
        pinpoint_lon = lon_m * 60 + lon_s

        if lat >= 0:
            index_i = num_rows - int(round(pinpoint_lat / arc_sec, 0)) - 1
        else:
            index_i = int(round(pinpoint_lat / arc_sec, 0))
        if lon >= 0:
            index_j = int(round(pinpoint_lon / arc_sec, 0))
        else:
            index_j = num_rows - int(round(pinpoint_lon / arc_sec, 0)) - 1

        lat_string = 'N' + str(int(lat_d)) if lat >= 0 else 'S' + str(abs(int(lat_d)))
        lon_string = 'E' + str(int(lon_d)).zfill(3) if lon >= 0 else 'W' + str(abs(int(lon_d))).zfill(3)  # has to be 080 format

        path = os.path.join(self.local_dir, lat_string + lon_string) + '.hgt'
        self.local_file_list.add(path)

        with open(path, "rb") as f:
            idx = (index_i * num_rows + index_j) * 2
            f.seek(idx)  # go to the right spot,
            buf = f.read(2)  # read two bytes and convert them:
            val = struct.unpack('>h', buf)  # ">h" is a signed two byte integer

            if not val == -32768:  # the not-a-valid-sample value
                return val[0]  # the raw result is (16, 0)
            else:
                return None

    def download(self) -> pd.DataFrame:

        for link in self.links:
            self.download_link(link)

        output = self.locations.copy()  # self.locations should be a data frame in this case
        result_set = []
        for idx, location in output[[self.lat_col, self.lon_col]].iterrows():
            location_ = [location[self.lat_col], location[self.lon_col]]
            result_set.append(self.look_up_altitude(*location_))

        output['ALTITUDE'] = result_set
        for file_ in self.local_file_list:
            os.remove(file_)
            zip_version = file_ + '.gz'
            os.remove(zip_version)
        return output


def download_altitude(locations, lat='LAT', lon='LON'):
    if isinstance(locations, pd.DataFrame):
        if lat not in locations or lon not in locations:
            raise KeyError('the dataframe must have a LAT column and a LON column')
        return AltitudeDownloader(lat_col=lat, lon_col=lon, locations=locations).download()
    elif isinstance(locations, tuple) or isinstance(locations, list):
        return AltitudeDownloader.from_list(locations).download()
    elif isinstance(locations, str) and len(locations) > 2:
        from ..data.util import get_lat_lon_from_string
        lat, lon = get_lat_lon_from_string(locations)
        return AltitudeDownloader.from_list([[lat, lon]]).download()
    else:
        raise TypeError('the locations must be one of the following three: tuple, list, dataframe')


