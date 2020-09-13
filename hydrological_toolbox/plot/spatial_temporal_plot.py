from names import get_first_name
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pandas as pd

from datetime import datetime
import logging
import tempfile
import os

from spatial_plot import SpatialPlot

logger = logging.getLogger(__name__)


class SpatialTemporalPlot:
    """
    generate a series of spatial snapshots for spatial temporal data.
    the snapshot supports plotting markers, circles and polygons, and any combination of them
    the output is a png panel image saved to the local directory
    """
    def __init__(self):
        self.canvass_list = {}
        self.graph_titles = None
        self._universal_elements = {}
        self.local_dir = tempfile.mkdtemp()

    def add_marker(self,
                   df_points: pd.DataFrame,
                   lat_col: str,
                   lon_col: str,
                   time_col: str,
                   **kwargs):
        """
        :param df_points: a dataframe of geopoints
        :param lat_col: the name of the latitude column
        :param lon_col: the name of the longitude column
        :param time_col: the name of the time column
        :param kwargs: any other kwargs specifies the setting for the folium marker
        :return:
        """
        for time_unit_name, df in df_points.groupby([time_col]):
            if time_unit_name not in self.canvass_list:
                self.canvass_list[time_unit_name] = SpatialPlot().add_marker(points=df,
                                                                             lat_col=lat_col,
                                                                             lon_col=lon_col,
                                                                             **kwargs)
            else:
                current_canvass = self.canvass_list[time_unit_name]
                current_canvass.add_marker(points=df, lat_col=lat_col, lon_col=lon_col, **kwargs)
        return self

    def add_circles(self,
                    df_points: pd.DataFrame,
                    lat_col: str,
                    lon_col: str,
                    value_col: str,
                    time_col: str,
                    **kwargs):
        """
        :param df_points: dataframe of geopoints
        :param lat_col: the name of the latitude column
        :param lon_col: the name of the longitude column
        :param time_col: the name of the time column
        :param kwargs: any other kwargs specifies the setting for the folium marker
        :param value_col: the name of value column that determines the size of the circle
        :return: SpatialTemporalPlot object
        """
        for time_unit_name, df in df_points.groupby([time_col]):
            if time_unit_name not in self.canvass_list:
                self.canvass_list[time_unit_name] = SpatialPlot().add_circles(points=df,
                                                                              lat_col=lat_col,
                                                                              lon_col=lon_col,
                                                                              y_col=value_col,
                                                                              **kwargs)
            else:
                current_canvass = self.canvass_list[time_unit_name]
                current_canvass.add_marker(points=df,
                                           lat_col=lat_col,
                                           lon_col=lon_col,
                                           value_col=value_col,
                                           **kwargs)
        return self

    def add_contour(self, df_contours, contour_col, time_col):
        for time_unit_name, df in df_contours.groupby([time_col]):
            if time_unit_name not in self.canvass_list:
                self.canvass_list[time_unit_name] = SpatialPlot().add_contour(df_contours[contour_col])
            else:
                current_canvass = self.canvass_list[time_unit_name]
                current_canvass.add_contour(df_contours[contour_col])
        return self

    def add_markers_for_all(self, df_points, lat_col, lon_col, **kwargs):
        """
        use this function if you wish to add markers for all figures across time
        """
        self._universal_elements['markers'] = (df_points, lat_col, lon_col, kwargs)
        return self

    def add_circles_for_all(self, df_points, lat_col, lon_col, value_col, **kwargs):
        """
        use this function if you wish to add the circles for all figures across time
        """
        self._universal_elements['circles'] = (df_points, lat_col, lon_col, value_col, kwargs)
        return self

    def add_contour_for_all(self, contour):
        """
        add a contour that applies for all figures in the time series
        """
        self._universal_elements['contour'] = contour
        return self

    def add_graph_titles(self, graph_titles):
        self.graph_titles = graph_titles
        return self

    def generate_image(self):
        """
        A generator for image and image names (if set by add_graph_titles)
        handles the following tasks related to image generation:
        1. if it's a universal contour (something applies to the whole image), add the same contour for each image
        2. look into the canvass dictionary and convert all canvass into images by doing screenshots
        3. determine the appropriate names for the images: the order is if there is a graph title list/dict
        that has been passed then use that else use the original values in the time column in the dataframe

        :return: iterator
        """
        # make sure to sort the time stamp so that early data show first
        time_stamp_in_order = sorted(self.canvass_list.keys())

        for key in self._universal_elements:
            for timestamp in self.canvass_list.keys():
                self.canvass_list[timestamp].add_contour(self._universal_elements[key])

        for idx, timestamp in enumerate(time_stamp_in_order):
            file_name = os.path.join(self.local_dir, str(timestamp) + '.png')
            self.canvass_list[timestamp].save_to_file(file_name)

            if self.graph_titles is not None and isinstance(self.graph_titles, dict):
                if timestamp in self.graph_titles:
                    graph_name = self.graph_titles[timestamp]
                else:
                    graph_name = timestamp
            elif self.graph_titles is not None and isinstance(self.graph_titles, list):
                graph_name = self.graph_titles[idx]
            else:
                graph_name = timestamp
            yield graph_name, mpimg.imread(file_name)

    def save_to_file(self, nrows: int, ncols: int, filename: str = None, dpi: int = None):
        """
        concatenate all the snapshots, convert to one composite png image, and save it to the current directory

        :param nrows: number of rows in the panel
        :param ncols: number of columns in the panel
        :param filename: the name of the file. if not specified, will be time_span + random human name
        :param dpi: resolution parameter for the image
        :return: None
        """
        font = {'weight': 'bold', 'size': 4}
        matplotlib.rc('font', **font)

        plt.figure()
        gs1 = gridspec.GridSpec(nrows=nrows, ncols=ncols)
        gs1.update(wspace=0.025, hspace=0.12)

        for i, (timestamp_or_name, img) in zip(range(nrows * ncols), self.generate_image()):
            ax1 = plt.subplot(gs1[i])
            plt.axis('off')
            plt.imshow(img)
            plt.title(str(timestamp_or_name))
            ax1.set_aspect('equal')

        if filename is None:
            # it's date time plus a random name
            time_stamp = str(datetime.now().month).zfill(2) + str(datetime.now().day)
            filename = '.'.join([time_stamp + '-' + get_first_name(), 'png'])

        logger.critical(f'the image has been saved as {filename} under {os.getcwd()}')
        if dpi is None:
            dpi = 500
        plt.savefig(filename, dpi=dpi)


if __name__ == '__main__':
    from data.util import get_state_contours
    from data import sample_data
    rain = sample_data.load_monthly_rain()
    rain = rain[rain.MONTH <= 4]
    rain_720 = rain.sample(720)[['LAT', 'LON', 'PRCP_MEAN', 'MONTH']]

    s = SpatialTemporalPlot()
    s.add_circles(df_points=rain,
                  lat_col='LAT',
                  lon_col='LON',
                  time_col='MONTH',
                  value_col='PRCP_MEAN',
                  multiplier=50,
                  line_color='#3186cc',
                  fill_color='#3186cc'
                  ).\
        add_contour_for_all(get_state_contours('SC')).\
        add_graph_titles(['January', 'February', 'March', 'April']).\
        save_to_file(nrows=1, ncols=4)

