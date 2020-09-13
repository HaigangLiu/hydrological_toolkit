from __future__ import annotations

import os
import time
import webbrowser
from typing import Union
import tempfile

from folium import Map, Marker, CircleMarker, GeoJson
from folium.plugins import MarkerCluster
from pandas import DataFrame, concat
from selenium import webdriver, common
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from data.util import get_state_contours

import logging
logger = logging.getLogger(__name__)


class SpatialPlot:
    """
    a plotting system that allows overlaying spatial elements on a map.
    Currently we support markers, circles and contours.
    """

    def __init__(self) -> None:
        # keep tabs on all the locations (self.data) that have been added and then determine the center of gravity
        # and then call fit_bounds from the map object to determine the best zoom size
        self.canvass = None
        self.filename_html = tempfile.NamedTemporaryFile().name + '.html'
        self.data = None

    def initialize_canvass(self, center=None) -> None:
        """
        determine the center based on a given locations by finding the latitude mean and longitude mean
        """
        if self.canvass is None:
            if center is None:
                self.canvass = Map(location=self.center, zoom_start=6)
            else:
                self.canvass = Map(location=center, zoom_start=6)

    def update_locations(self, new_data, lat_col, lon_col):
        """
        since we allow multiple calls to add contour, points and markers, it's important to keep all locations
        in track. Doing so is helpful to, for example, determine the boundaries for the whole graph.
        """
        new_data = new_data[[lat_col, lon_col]]
        new_data.columns = ['LAT', 'LON']
        self.data = concat([self.data, new_data])

    @property
    def center(self):
        if self.data is not None:
            df_points = self.data[['LAT', 'LON']]
            center = df_points.mean(axis=0).values
            center = [round(location, 2) for location in center]
            return center
        else:
            raise ValueError('no location data are available.')

    def add_marker(self,
                   points: Union[DataFrame, list],
                   lat_col: str = 'LAT',
                   lon_col: str = 'LON',
                   clustered: str = 'auto',
                   **kwargs) -> SpatialPlot:
        """
        this method adds a marker on the map. All info besides the locations will be neglected.
        allows either list of list or data frame as input

        The clustered gives an option to fold the locations into clusters if there are too many of
        them. This functionality will be activated when there is more than 30 locations. User can
        overwrite this behavior by setting clustered to 'no'.

        :param points: either a dataframe or a list with location info and the values observed at the location
        :param lat_col: the column name (str) for latitude information
        :param lon_col: the column name (str) for longitude information
        :param clustered: determines if we fold the markers to clusters.
            'auto' will fold the markers if there's too many (30).
            'no' means we won't fold the markers regardless.
            'yes' means we will fold the markers regardless.
        """
        if clustered not in ['yes', 'no', 'auto']:
            raise ValueError('clustered only take yes, no or auto as legit input values')

        if isinstance(points, DataFrame):
            if lat_col not in points or lon_col not in points:
                raise KeyError('We need a column for latitude information and a column for longitude')
            df_points = points

        elif isinstance(points, list):
            logger.critical('we will assume the first position is latitude, and the second is longitude')
            df_points = DataFrame(points).iloc[:, 0:2]
            df_points.columns = [lat_col, lon_col]

        else:
            raise TypeError('only list and dataframe are supported in this method')

        self.update_locations(new_data=df_points, lon_col=lon_col, lat_col=lat_col)
        self.initialize_canvass()

        use_cluster, marker_cluster = False, None
        if (clustered == 'auto' and len(df_points) >= 30) or (clustered == 'yes'):
            marker_cluster = MarkerCluster(icons="dd").add_to(self.canvass)
            use_cluster = True
            logger.critical(f'there are {len(df_points)} locations, and we fold them by clustering. '
                            f'Set clustered to no to overwrite this behavior.')

        for _, point in df_points.iterrows():
            geo_location = [point[lat_col], point[lon_col]]
            if use_cluster:
                Marker(location=geo_location, **kwargs).add_to(marker_cluster)
            else:
                Marker(location=geo_location, **kwargs).add_to(self.canvass)
        return self

    def add_circles(self,
                    points: Union[DataFrame, list],
                    lat_col: str,
                    lon_col: str,
                    y_col: str,
                    multiplier: float = 2,
                    **kwargs) -> SpatialPlot:
        """
        the difference between the add_circle and add_marker is that the former uses the size of the circle
        to visualize the value observed at that point
        :param points: either a dataframe or a list with location info and the values observed at the location
        :param lat_col: the column name (str) for latitude information
        :param lon_col: the column name (str) for longitude information
        :param y_col: the column name (str) for observed value at that location
        :param multiplier: size of the point. Increase this value to get large circles.
        """
        if isinstance(points, DataFrame):
            if lat_col not in points or lon_col not in points or y_col not in points:
                raise KeyError('We need a column for latitude information and a column for longitude')
            df_points = points

        elif isinstance(points, list):
            logger.critical('we will assume the first position is latitude,'
                            ' and the second is longitude, and the third position is y')
            df_points = DataFrame(points).iloc[:, 0:3]
            df_points.columns = [lat_col, lon_col, y_col]
        else:
            raise TypeError('only list and dataframe are supported in this method')

        df_points = df_points[[lat_col, lon_col, y_col]]

        self.update_locations(new_data=df_points, lat_col=lat_col, lon_col=lon_col)
        self.initialize_canvass()

        for idx, row in df_points.iterrows():
            value = row[y_col]
            location = (row[lat_col], row[lon_col])
            CircleMarker(location=location,
                         radius=multiplier*abs(value),
                         **kwargs) \
                .add_to(self.canvass)
        return self

    def add_contour(self, contour: Union[Polygon, MultiPolygon, str]) -> SpatialPlot:
        """
        :param contour: either a polygon object or a string (state acronym)
        if it's a string, this method will pull the contour with get_state_contour() function
        and plot it.
        """
        if isinstance(contour, str):
            contour_polygon = get_state_contours(contour)
        elif isinstance(contour, MultiPolygon) or isinstance(contour, Polygon):
            contour_polygon = contour
        else:
            raise TypeError('this method only supports state name or polygon object from shapely')

        min_x, min_y, max_x, max_y = contour_polygon.bounds
        make_df = DataFrame({'LAT': [min_y, max_y], 'LON': [min_x, max_x]})
        self.update_locations(new_data=make_df, lat_col='LAT', lon_col='LON')

        [centroid_] = list(contour_polygon.centroid.coords)
        longitude, latitude = centroid_
        self.initialize_canvass(center=[latitude, longitude])  # no need to infer the center this time just use centroid

        GeoJson(contour_polygon).add_to(self.canvass)
        return self

    def _rebalance(self):
        """
        write the map object to a local html so we can open it.
        before saving: we need to decide the best way to show the map

        More details:
        folium package has a parameter the initial zoom (zoom_start). this function
        helps picking the best zoom size; since we would like to support saving to local
        functionality (by doing screen shot), it is important to know the best zoom scale
        """
        sw = self.data[['LAT', 'LON']].min().values.tolist()
        ne = self.data[['LAT', 'LON']].max().values.tolist()
        self.canvass.fit_bounds([sw, ne])
        self.canvass.save(self.filename_html)

    def show(self):
        """
        show() will open the map in a browser
        the map will be automatically saved to the current directory with a time stamp
        """
        self._rebalance()
        webbrowser.open('file://' + os.path.realpath(self.filename_html))
        logger.critical(f'the graph has been saved into file named {self.filename_html}')

    def save_to_file(self, filename=None):
        """
        save the html file to a local dir. The implementation is simply open a browser, take a screen shot
        and then close the browser.
        """
        self._rebalance()
        try:
            browser = webdriver.Firefox()
            browser.get('file://' + os.path.realpath(self.filename_html))
            time.sleep(1)
            if filename is None:
                # it's date time plus a random name
                from datetime import datetime
                from names import get_first_name
                time_stamp = str(datetime.now().month).zfill(2) + str(datetime.now().day)
                filename = '.'.join([time_stamp + '-' + get_first_name(), 'png'])
            browser.save_screenshot(os.path.join(os.getcwd(), filename))
            browser.quit()
        except common.exceptions.WebDriverException:
            warning_message = """
            if you are running into a geckodriver issue. Make sure you have firefox installed
            on a mac os system: make sure to run brew install geckodriver in the terminal
            or you can do pip install selenium==2.53.6
            """
            logger.critical(warning_message)


if __name__ == '__main__':

    from data import sample_data
    flood = sample_data.load_monthly_flood()
    flood.drop_duplicates(['LAT', 'LON'], inplace=True)

    m = SpatialPlot()
    m.add_contour('SC').\
        add_marker(flood,
                   lat_col='LAT',
                   lon_col='LON',
                   clustered='auto').\
        save_to_file()

    exit()



    import numpy as np
    sc_ = get_state_contours('SC')
    lat = np.arange(33, 35, 0.3)
    lon = np.arange(-81.5, -80, 0.3)

    lat_, lon_ = np.meshgrid(lat, lon)
    lat_ = lat_.reshape(-1)
    lon_ = lon_.reshape(-1)

    simulated_residuals = np.random.standard_normal(size=len(lat_))
    simulated_residuals = list(zip(lat_, lon_, simulated_residuals * 5))
    simulated_residuals = DataFrame(simulated_residuals, columns=['lat', 'lon', 'value'])

    positives = simulated_residuals[simulated_residuals['value'] >= 0]
    negatives = simulated_residuals[simulated_residuals['value'] < 0]

    m = SpatialPlot()
    m.add_contour('SC'). \
        add_circles(points=positives,
                    lat_col='lat',
                    lon_col='lon',
                    y_col='value',
                    color='red'). \
        add_circles(points=negatives,
                    lat_col='lat',
                    lon_col='lon',
                    y_col='value',
                    color='blue').\
        save_to_file()

    # m.add_contour('sc'). \
    #     add_circles(points=points_, lat_col='lat', lon_col='lon', y_col='value'). \
    #     add_marker(points=points_, lat_col='lat', lon_col='lon').show()

