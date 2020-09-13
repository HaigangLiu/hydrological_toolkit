from pandas import DataFrame, Series
from scipy.spatial import cKDTree as KDtree
from model.base_model import Model


class InverseDistanceModel(Model):
    """
    Make predictions based on nearby values.
    User can use the number of neighboring points as threshold
    p = 2 gives Euclidean distance 1 is Manhattan distance for example
    """
    def __init__(self, threshold: int = 3, p: float = 2.0):
        super().__init__()

        if not isinstance(threshold, int):
            raise TypeError('threshold needs to be an integer because it is the number of neighbors to include')
        if threshold < 0:
            raise ValueError('the number of neighbors around must be a positive number')
        self.threshold = threshold
        self.p = p

    def fit(self, df, location_cols, y_col, **kwargs):
        super().fit(df, location_cols, y_col)
        self.model = KDtree(self.locations)

    def _predict_single_loc(self, new_loc, num_neighbors):
        dist, index = self.model.query(new_loc, [num_neighbors])
        if 0 in dist:
            arg_max, *_ = index
            return self.y[arg_max]

        weights = 1/(dist**self.p)
        standardized_weights = weights/sum(weights)
        prediction = sum(standardized_weights*self.y[index])
        return prediction

    def predict(self, df, location_cols=None, **kwargs):
        super().predict(df, location_cols=location_cols)
        return DataFrame(self.locations_new).apply(self._predict_single_loc,
                                                   axis=1,
                                                   num_neighbors=self.threshold)


if __name__ == '__main__':

    import numpy as np
    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    z = np.random.standard_normal(100)

    training = DataFrame([x, y, z])
    training = training.T
    training.columns = ['x', 'y', 'z']

    x_star = np.random.standard_normal(100000)
    y_star = np.random.standard_normal(100000)
    z_star = np.random.standard_normal(100000)

    test = DataFrame([x_star, y_star, z_star])
    test = test.T
    test.columns = ['x', 'y', 'z']

    m = InverseDistanceModel()
    m.fit(df=training, location_cols=['x', 'y'], y_col=['z'])
    result = m.predict(df=training)

    print(result)



