from abc import ABC, abstractmethod
from pandas import DataFrame


# note: by convention, fit means find parameters best fits the data, hence one should provide data at "fit" step.
# this is also how sklearn does it. In __init__ step we only provide model-related params


class Model(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.y = None
        self.X = None
        self.locations = None
        self._name_location_cols = None

        self.X_new = None
        self.locations_new = None
        self.y_pred = None

    @abstractmethod
    def fit(self, df, location_cols, y_col, **kwargs):
        """
        location and response variables are the bare minimum
        """
        if not isinstance(df, DataFrame):
            raise TypeError(f'currently, we only support pandas data frame as model input type.')

        for location_col in location_cols:
            if location_col not in df:
                raise KeyError(f'{location_col} is not in the data frame')

        if isinstance(y_col, list):
            if len(y_col) == 1:
                [y_col] = y_col
            else:
                raise ValueError('the response variable can only be one-dimensional')

        if y_col not in df:
            raise KeyError(f'{y_col} is not in the data frame')

        self.y = df[y_col].values.flatten()
        self.locations = df[location_cols].values
        self._name_location_cols = location_cols  # cache this because we might want use it in predict()

    @abstractmethod
    def predict(self, df, location_cols=None, **kwargs):
        """these are the bare minimum for any sub class"""
        if not isinstance(df, DataFrame):
            raise TypeError(f'currently, we only support pandas data frame as model input type.')

        if location_cols is None:
            location_cols = self._name_location_cols

        for col in location_cols:
            if col not in df:
                raise KeyError(f'the column {col} with location information is not in the dataframe')

        self.locations_new = df[location_cols].values
