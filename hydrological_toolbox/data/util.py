import json
from scipy.spatial import KDTree
from typing import List, Union
import logging
from pkg_resources import resource_filename
from functools import cached_property

from pandas import date_range, DataFrame, concat, merge, read_parquet
import numpy as np
import fiona
from geopy.geocoders import Nominatim
import requests
from shapely.geometry import shape, MultiPolygon
import shapely.ops
from scipy.spatial import distance_matrix
import pyproj

logger = logging.getLogger(__name__)

states_name_and_patch_json = resource_filename('hydrological_toolbox', 'asset.state_names.json')
state_patches = resource_filename('hydrological_toolbox', 'asset.state_boundaries.json')
state_index_in_json = resource_filename('hydrological_toolbox', 'asset.mapping_state_to_contour_index.json')
state_contours = resource_filename('hydrological_toolbox', 'asset.state_shape.cb_2017_us_state_500k.shp')


def convert_coordinates(df: DataFrame,
                        lat: str = 'LAT',
                        lon: str = 'LON',
                        ) -> DataFrame:
    """
    Assuming that earth is a perfect sphere, and convert the lat and lon into a 3D vector.
    The radius of earth is 3959
    """
    R: int = 3959
    if lat not in df or lon not in df:
        raise KeyError(f'make sure the data frame has both a column for latitude and a column of longitude')

    lat_r, lon_r = np.radians(df[lat]), np.radians(df[lon])
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)

    output = DataFrame(np.array(list(zip(x, y, z))))
    output.columns = ['x', 'y', 'z']
    return output


def get_state_name(state: str):
    """
    translate the state acronym into the spelled out version
    the database also includes us small islands.
    >>> get_state_name('SC')
    'South Carolina'
    >>> get_state_name('PR')
    'Puerto Rico'
    >>> get_state_name('HA')
    Traceback (most recent call last):
        ...
    KeyError: 'this state acronym does not exist.'
    >>> get_state_name('South Carolina')
    Traceback (most recent call last):
        ...
    ValueError: the input should be an acronym of a US state.
    """
    if len(state) != 2:
        raise ValueError('the input should be an acronym of a US state.')

    state = state.upper()
    with open(states_name_and_patch_json) as json_file:
        state_names = json.load(json_file)
        if state not in state_names:
            raise KeyError('this state acronym does not exist.')
        return state_names[state]


def get_state_contours(state_acronym: str, lat_first: bool = False) -> MultiPolygon:
    """
    get the contour information for the give state name
    Note: each state has an id in the cb_2017_us_state_500k.shp file. The mapping from name to id is stored in
    mapping_state_to_contour_index.json. User only has to provide the state acronym, such as SC.

    Note:
        in the file we obtain from nws, longitude comes as the first dimension. Users can change this by setting
        lat_first=True
        lat_first=True
    """
    state_name = get_state_name(state_acronym)
    with open(state_index_in_json) as json_file:
        state_index = json.load(json_file)
        if state_name not in state_index:
            raise KeyError('we do not have the contour information for the given state')
        else:
            state_id = int(state_index[state_name])

    shape_for_states = fiona.open(state_contours)
    state_contour = shape(shape_for_states[state_id]['geometry'])
    if lat_first is True:
        state_contour = shapely.ops.transform(lambda x, y: (y, x), state_contour)
    else:
        logger.critical(f'by default the longitude comes first, one can flip it by setting flip_lat_lon=True')
    return state_contour


def get_in_between_dates(start_date: str, end_date: str) -> List:
    """
    The input and output are consistent. If start date and end date are both '1900-01-01' the output would also
    be in days. Otherwise, if it is like '1990-01' then the output will be month.

    >>> get_in_between_dates('1990-01-01', '1990-01-03')
    ['1990-01-01', '1990-01-02', '1990-01-03']
    >>> get_in_between_dates('1990-01', '1990-03')
    ['1990-01', '1990-02', '1990-03']
    >>> get_in_between_dates('1990-01', '1990-01-20')
    Traceback (most recent call last):
        ...
    ValueError: the start date and end date should be of equal length
    """
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise TypeError('both start date and input have to be of string type')

    if len(start_date) != len(end_date):
        raise ValueError('the start date and end date should be of the same format and the same length')

    if len(start_date) == 7:  # monthly
        # MS means month start to make sure the interval we get is both inclusive
        list_of_dates = date_range(start_date, end_date, freq='MS').to_list()
        return [str(time_stamp)[0:7] for time_stamp in list_of_dates]

    elif len(start_date) == 10:  # daily
        list_of_dates = date_range(start_date, end_date, freq='D').to_list()
        return [str(time_stamp)[0:10] for time_stamp in list_of_dates]

    else:
        raise ValueError('only accept format like 1990-01-01 or 1990-01.')


def get_state_patch(state_acronym):
    """
    find maximum and minimum of latitude and longitude for a given state, a rectangle patch that covers the state.
    >>> get_state_patch('SC')
    {'name': 'South Carolina', 'min_lat': 32.0453, 'max_lat': 35.2075, 'min_lng': -83.3588, 'max_lng': -78.4836}
    >>> get_state_patch('NY')
    {'name': 'New York', 'min_lat': 40.4772, 'max_lat': 45.0153, 'min_lng': -79.7624, 'max_lng': -71.7517}
    >>> get_state_patch('HA')
    Traceback (most recent call last):
        ...
    KeyError: 'We do not have the max/min location information for the given state. Make sure the acronym is accurate'
   """

    state_acronym = state_acronym.upper()
    if len(state_acronym) != 2:
        raise ValueError('The input should be a two letter acronym for a state')

    with open(state_patches) as state_boundary_js:
        patch_for_states = json.load(state_boundary_js)

    if state_acronym not in patch_for_states:
        raise KeyError('We do not have the max/min location information for the given state. '
                       'Make sure the acronym is accurate')
    return patch_for_states[state_acronym]


def convert_daily_to_monthly(df: DataFrame,
                             coordinate_x: str,
                             date_col: str,
                             value_col: str,
                             coordinate_y: str = None,
                             agg_funcs: Union[List, dict] = None):
    """
    the original data is daily. use this function in case monthly data is needed.
    coordinate_x and y column can be anything that uniquely determines the location of a point spatially
    value_col is the column for rainfall, for example
    Note 1:
        agg_function should be the input of pandas.groupby.agg(agg_func)
    Note 2:
        we can use coordinate_x and y as longitude and latitude pair to uniquely define a location/station.
        Alternatively, one can use the station id to identify a spatial location. If so, coordinate_x will be
        the station id column, and coordinate_y will be None
    """
    try:
        df['year_and_month'] = df[date_col].apply(lambda x: x[0:7])
    except TypeError:  # input type may be time stamp
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['year_and_month'] = df.apply(lambda row: '-'.join([str(row['year']), str(row['month'])]), axis=1)
        df.drop(columns=['year', 'month'], inplace=True)

    except ValueError:
        logger.critical('the date column does not have a yyyy-mm-dd format')
        return None

    if agg_funcs is None:
        # can also use built in funcs min max, but the np version performs better
        agg_funcs = {value_col: [('MIN', np.min),
                                 ('MAX', np.max),
                                 ('MEAN', np.mean),
                                 ('SUM', np.sum)]}
    if coordinate_y is None:
        # in some cases the identifier can have only one dimension, like station id
        grouper = [coordinate_x, 'year_and_month']
    else:
        grouper = [coordinate_x, coordinate_y, 'year_and_month']
    df = df.groupby(grouper).agg(agg_funcs)
    df.reset_index(drop=False, inplace=True)
    year_and_month = df.year_and_month.str.split('-', expand=True)
    year_and_month = year_and_month.applymap(int)
    year_and_month.columns = ['YEAR', 'MONTH']

    output = concat([df, year_and_month], axis=1)
    new_names = []
    # this for loop is to handle the consequences of messy multi-level index
    for col in output.columns:
        if isinstance(col, str):
            new_names.append(col)
        else:
            new_names.append(' '.join(col).strip().replace(' ', '_'))
    output.columns = new_names
    output.drop(columns=['year_and_month'], inplace=True)
    output.sort_values(['YEAR', 'MONTH'], inplace=True)
    return output


def project(shapely_object_or_list,
            original_prj=None,
            new_prj=None):
    """
    :param shapely_object_or_list: The input can be a shapely object or a list of them.
     If input is a list then the output would also a list of re-projected shapely object.
    :param original_prj: e.g., epsg: 4326
    :param new_prj: e.g., epsg: 4326.
    :return: (a list of) shapely project.
    """
    original_prj = 'epsg:4326' if original_prj is None else original_prj
    new_prj = 'epsg:3857' if new_prj is None else new_prj
    transformer = pyproj.Transformer.from_crs(crs_from=original_prj, crs_to=new_prj)

    if isinstance(shapely_object_or_list, list):
        converted_shapely_list = []
        for shapely_object in shapely_object_or_list:
            converted_shapely_list.append(shapely.ops.transform(transformer.transform, shapely_object))
        return converted_shapely_list
    return shapely.ops.transform(transformer.transform, shapely_object_or_list)


def project_deprecated(contour,
                       original_prj=None,
                       new_prj=None):
    """
    This helper function converts one set of projection system into another.
    Note:
    This method has been deprecated due to the API change from pyproj developer.
    And this method takes (lon, lat) format, which is also different from the newer API.
    """
    original_prj = 'epsg:4326' if original_prj is None else original_prj
    new_prj = 'epsg: 3857' if new_prj is None else new_prj

    def transform_op(original, new):
        return pyproj.transform(pyproj.Proj(init=original_prj),
                                pyproj.Proj(init=new_prj),
                                original,
                                new)

    reprojected = shapely.ops.transform(transform_op, contour)
    return reprojected


def spatial_left_join(df_left: DataFrame,
                      df_right: DataFrame,
                      left_on: Union[str, list],
                      right_on: Union[str, list],
                      how: str = 'left') -> DataFrame:
    """
    merge left with right table based on spatial proximity.
    we will start with left data frame and for each data in left,
    we find the closest point in the right.

    the df_left_coord and df_right_coord must uniquely determines the row in
    df_left and df_right respectively.
    Meaning that, if the input is a spatial temporal data frame, one needs to
    group by date to make sure the locations are unique.
    """
    # check if valid
    if len(df_left[left_on].drop_duplicates()) != len(df_left):
        raise ValueError('the locations in the left data frame contains duplicates.')

    if len(df_right[right_on].drop_duplicates()) != len(df_right):
        raise ValueError('the locations in the right data frame contains duplicates')

    df_right_coord_ = []
    # this is to avoid the case where both left and right have the same col names
    # pandas will raise an error when we try to join
    for dim in right_on:
        if dim in left_on:
            df_right_coord_.append(dim + '_RIGHT')
        else:
            df_right_coord_.append(dim)

    if isinstance(left_on, list) and isinstance(right_on, list):
        new_columns = left_on.copy()
        new_columns.extend(df_right_coord_)
    elif isinstance(left_on, str) and isinstance(left_on, str):
        new_columns = [left_on, df_right_coord_]
    else:
        raise TypeError('only accept string or list as the coord columns')

    df_change_right_name = {old_name: new_name for old_name, new_name
                            in zip(right_on, df_right_coord_)}
    df_right = df_right.rename(df_change_right_name, axis=1)

    distance_ = distance_matrix(df_left[left_on].values,
                                df_right[df_right_coord_].values)

    min_index_for_each_left_table_location = np.argmin(distance_, axis=1)
    df_right_filtered = df_right.iloc[min_index_for_each_left_table_location].reset_index(drop=True)

    middle_step_with_both_locations = concat([df_left.reset_index(drop=True)[left_on],
                                              df_right_filtered.reset_index(drop=True)[df_right_coord_]], axis=1)

    middle_step_with_both_locations.columns = new_columns
    output = merge(df_left, middle_step_with_both_locations, on=left_on, how=how)
    output = merge(output, df_right, on=df_right_coord_, how=how)
    return output


def spatial_left_join_k_neighbors(df_left: DataFrame,
                                  df_right: DataFrame,
                                  left_on: Union[str, list],
                                  right_on: Union[str, list],
                                  k: int = 4,
                                  forbid_duplicates: bool = True,
                                  include_distance: bool = True) -> DataFrame:
    """
    merge left with right table based on spatial proximity.
    we will start with left data frame and for each data in left,
    we find the closest k point in the right.

    Note that we use the partition algorithm under the hood and thus the
    LAT_RIGHT_0 is not necessary the nearest one, but one of the nearest k neighbors.
    """
    # check if valid
    if forbid_duplicates:
        if len(df_left[left_on].drop_duplicates()) != len(df_left):
            raise ValueError('the locations in the left data frame contains duplicates.')

        if len(df_right[right_on].drop_duplicates()) != len(df_right):
            raise ValueError('the locations in the right data frame contains duplicates')

    distance_ = distance_matrix(df_left[left_on].values, df_right[right_on].values)  # (192, 5103)
    min_index_for_each_left_table_location = np.argpartition(distance_, k, axis=1)[:, 0: k]

    arg_min_k, all_k_neighbors = k, []

    for arg_min in range(arg_min_k):
        new_col_names_with_k = []
        indexer = min_index_for_each_left_table_location[:, arg_min]
        df_right_filtered = df_right.iloc[indexer]

        for col_k in df_right_filtered.columns:
            new_col_names_with_k.append(col_k + '_' + 'RIGHT' + '_' + str(arg_min))
        df_right_filtered.columns = new_col_names_with_k
        df_right_filtered.reset_index(drop=True, inplace=True)
        all_k_neighbors.append(df_right_filtered)

    all_k_neighbors.insert(0, df_left)

    if include_distance:
        distance_array = np.take_along_axis(distance_, min_index_for_each_left_table_location, axis=1)
        distance_array_names = ['DISTANCE' + '_' + str(i) for i in range(k)]
        distance_df = DataFrame(distance_array)
        distance_df.columns = distance_array_names
        all_k_neighbors.append(distance_df)

    output = concat(all_k_neighbors, axis=1)
    return output


class GridMaker:
    def __init__(self,
                 df: Union[DataFrame, None] = None,
                 lat_col: str = 'LAT',
                 lon_col: str = 'LON',
                 step_lat: float = 0.1,
                 step_lon: float = 0.1):

        self._df = df
        self.lat = lat_col
        self.lon = lon_col
        self.step_lat = step_lat
        self.step_lon = step_lon

    @classmethod
    def from_tuple(cls, tuples, step_lat, step_lon, lat_first=True):
        logger.critical('We will assume latitude is the first value in the tuple'
                        ' and longitude comes second. Set lat_first=False to change this')

        if not isinstance(tuples[0], list) and isinstance(tuples[0], tuple):
            raise TypeError('A two dimensional tuple/list is expected in this case')

        lat_col, lon_col = 'LAT', 'LON'
        df_input = DataFrame(tuples)

        rows, cols = df_input.shape

        if cols > 2:
            df_input = df_input.iloc[:, 0:2]
            logger.critical('Only kept the first 2 columns, namely lat and lon')
        elif cols <= 1:
            raise ValueError('there is less than or equal to 1 column in the data first.'
                             'at least 2 needed.')

        df_input.columns = [lat_col, lon_col] if lat_first else [lon_col, lat_col]

        return cls(df=df_input,
                   lat_col=lat_col,
                   lon_col=lon_col,
                   step_lat=step_lat,
                   step_lon=step_lon)

    from_list = from_tuple

    @cached_property
    def df(self):
        if not isinstance(self._df, DataFrame):
            raise TypeError('only support pandas dataframe. '
                            'Use .from_tuple or from_list if your input is a different type.')
        if self.lat not in self._df or self.lon not in self._df:
            raise KeyError(f'the data frame either misses {self.lat} or {self.lon}')
        return self._df

    @cached_property
    def lat_max(self):
        return self.df[self.lat].max()

    @cached_property
    def lat_min(self):
        return self.df[self.lat].min()

    @cached_property
    def lon_max(self):
        return self.df[self.lon].max()

    @cached_property
    def lon_min(self):
        return self.df[self.lon].min()

    def generate_grid(self):
        lat_range = np.arange(start=self.lat_min,
                              stop=self.lat_max,
                              step=self.step_lat)
        lon_range = np.arange(start=self.lon_min,
                              stop=self.lon_max,
                              step=self.step_lon)
        lat_, lon_ = np.meshgrid(lat_range, lon_range)
        grid = np.vstack([lat_.reshape(-1), lon_.reshape(-1)])
        return lat_range, lon_range, grid

    def __call__(self,
                 df: DataFrame,
                 lat_col: str = 'LAT',
                 lon_col: str = 'LON',
                 step_lat: float = 0.1,
                 step_lon: float = 0.8):

        self._df = df
        self.lat = lat_col
        self.lon = lon_col
        self.step_lat = step_lat
        self.step_lon = step_lon
        return self.generate_grid()


def get_lat_lon_from_string(address):
    geolocator = Nominatim(user_agent="city")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude


get_lat_lon_from_address = get_lat_lon_from_string


class SiteInfoDownloader:
    def __init__(self, list_of_sites):
        self.result = {}
        self.list_of_sites = list_of_sites

    def run(self):
        for site in self.list_of_sites:
            self.result.update(self.get_single_site_info(site))
        return self.result

    @classmethod
    def from_string(cls, site):
        list_of_sites = [site]
        return cls(list_of_sites)

    @staticmethod
    def get_single_site_info(site_number: str):
        """
        if you got a site number, this function can help you to get the latitude, longitude and description of that site
        :param site_number: str: a unique identifier for a usgs site.
        Must be a string since some sites start with 0
        :return: a dictionary of lat and lon and a string description of the place.
        """
        if isinstance(site_number, int):
            raise TypeError('site number must be a string since some start with 0')

        template = f'https://waterdata.usgs.gov/nwis/nwismap?site_no={site_number}&format=sitefile_output' \
                   f'&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm' \
                   f'&column_name=dec_lat_va&column_name=dec_long_va'
        site_info = {}
        site = requests.get(template)

        if site.ok:
            for line in site.text.split('\n'):
                if line.startswith('USGS'):
                    _, site_no, description, lat, lon, _, _ = line.split('\t')
                    site_info[site_no] = (description, lat, lon)
            if not site_info:
                logger.critical(f'did not find anything at site {site_number}. verify the site number again')
        else:
            logger.critical(f'cannot establish connection for site {site_number}')
        return site_info


def get_site_info(site):
    """
    if you got a site number, this function can help you to get the latitude, longitude and description of that site
    the input could be either a site number or a list of site number
    """
    if isinstance(site, str):
        return SiteInfoDownloader.from_string(site).run()
    elif isinstance(site, list):
        return SiteInfoDownloader(site).run()
    else:
        raise TypeError('only two input type supported: string and list')


def get_x_y_projections(locations, mapping=None):
    if not mapping:
        mapping = read_parquet('../asset/new_rain_mapping.parquet')
    tree = KDTree(mapping[['lat', 'lon']])
    list_of_distances, list_of_indices = tree.query(locations)
    return mapping.iloc[list_of_indices]
# user can call this function directly without interacting explicitly
# with initiating the class
# generate_grid = GridMaker()
