import os
import theano
import pymc3 as pm
import numpy as np
from typing import Union, List
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
from PIL import Image
from base_model import BaseModel
from .helper import summarize_input

theano.config.compute_test_value = "ignore"  # no default value error shall occur without this line


class CarModel(BaseModel):
    """
    Fit a conditional autoregressive model (spatial-temporal model).
    To fit a pure spatial model, use spatial_model module
    Args:
        response (np.array): 1-d array for response variable
        location_var: 1-d array to store location information
        covariates: nd array to store covariates;
            intercepts will be generated automatically.
    """

    def __init__(self,
                 response: np.ndarray,
                 locations: np.ndarray,
                 covariates: Union[None, List[List[str]]] = None,
                 autoreg: int = 1):

        super().__init__(response, locations, covariates)
        self.N, self.number_of_days = response.shape

        # build the covariate array and the proximity matrix
        if covariates is None:
            self.covariates = [np.ones_like(response)]
        self.covariates = self.pad_covariates(self.covariates, self.number_of_days)
        self.adjacency_matrix, self.weight_matrix = self.get_weight_matrices(locations)
        self.residuals = None
        summarize_input(size=self.N, days=self.number_of_days, covars=len(self.covariates))

        # build additional stuff for autoregressive version of this model
        self.shifted_response = []
        self.autoreg = autoreg

        if autoreg:  # redefine both x and y
            covariates_auto = []  # new X
            if self.covariates:
                for covariate in self.covariates:
                    covariate_remove_extra_days = covariate[:, autoreg:]
                    covariates_auto.append(covariate_remove_extra_days)
                self.covariates = covariates_auto
                del covariates_auto

            for i in range(autoreg):  # new Y_{t-1}
                right = self.number_of_days - autoreg + i
                make_shifted_y = self.response[:, i:right]
                self.shifted_response.append(make_shifted_y)

            self.response = self.response[:, autoreg:]
            self.number_of_days = self.number_of_days - autoreg
            print(f'autoregressive term is {autoreg}, and first {autoreg} day(s) will be used as covariates')


    @classmethod
    def from_pandas(cls, dataframe, response, locations, covariates=None, **kwargs):
        """
        initialize the model with data frame and column names
        response (list): name of the response variable column (should be more than one since there is one
            for each day/month/year)
        locations (list): list of names of columns that specifies location
        covariates (list, NoneType by default): list of names of columns that specifies the covariates.
            - the format has to be lists of list in case there are multiple variables for multiple days.
            - e.g. [['rain_day_1', 'rain_day_2'], ['flood_day_1', flood_day_2']
        """
        if len(response) == 1:
            raise ValueError("the response variable should be e.g. ['y_day1', 'y_day2', ...]")
        if type(locations) != list:
            raise TypeError('Locations has to be a list of column names that determines the location')

        if covariates:
            if not isinstance(covariates, list):
                raise TypeError('The covariates names must be a list')

            if not isinstance(covariates[0], list):
                raise TypeError("the covariates list should be a list of list: e.g."
                                "[['rain_day_1', 'rain_day_2'], ['flood_day_1', flood_day_2']]")

        cov_list = None
        if covariates:
            cov_list = []
            for cov in covariates:
                cov_list.append(dataframe[cov])

        response_input = dataframe[response].values
        locations_input = dataframe[locations].values

        return cls(response_input, locations_input, cov_list, **kwargs)

    def pad_covariates(self, covariates, expected_dim, intercept=False):
        """
        a helper function to make sure the dimension of input numpy arrays are correct
        three situations are considered:
         1. shape = (10, ): covert to (10, 1) then populate to (10,10)
            where 10 is the number of days
        2. shape = (10 ,1), like first case, populate to (10, 10)
        3. shape = (10, 9) raise ValueError since data type not understood.
        """
        covariates_ = []
        for covariate in covariates:
            if np.ndim(covariate) == 1:
                covariate = np.tile(covariate[:, None], expected_dim)
            elif np.ndim(covariate) == 2:
                if covariate.shape[1] == expected_dim:
                    pass
                elif covariate.shape[1] == 1:
                    covariate = np.tile(covariate, expected_dim)
                else:
                    raise ValueError(f'the proper dimension is {expected_dim}, get {covariate.shape[1]} instead')
            else:
                raise ValueError('dimension of covariates can only be either 1 or 2')
            covariates_.append(covariate)
        if intercept:
            covariates_.append(np.ones((self.N, expected_dim)))
        return covariates_

    @staticmethod
    def get_weight_matrices(location_arr):
        try:
            location_list = location_arr.tolist()
        except AttributeError:
            print('all inputs must be numpy.array type')
            return None

        neighbor_info = []
        weight_info = []
        proximity_info = []

        for entry in location_list:
            w = [0 if entry != comp else 1 for comp in location_list]
            neighbor_info.append(w)
        neighbor_info = np.array(neighbor_info)

        for idx, row in enumerate(neighbor_info):
            mask = np.argwhere(row == 1).ravel().tolist()
            mask.remove(idx)  # delete the location itself.
            proximity_info.append(mask)
            weight_info.append([1] * len(mask))

        N = len(location_arr)
        weight_matrix = np.zeros((N, N))
        adjacency_matrix = np.zeros((N, N), dtype='int32')

        for i, a in enumerate(proximity_info):
            adjacency_matrix[i, a] = 1
            weight_matrix[i, a] = weight_info[i]
        return [adjacency_matrix, weight_matrix]

    def fit(self, sample_size=3000, sig=0.95):
        with pm.Model() as self.model:
            # Priors for spatial random effects
            tau = pm.Gamma('tau', alpha=2., beta=2.)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            D = np.diag(self.weight_matrix.sum(axis=1))
            self.phi = pm.MvNormal('phi',
                                   mu=0,
                                   tau=tau * (D - alpha * self.weight_matrix),
                                   shape=(self.number_of_days, self.N)  # the second dim has covar structure
                                   )
            mu_ = self.phi.T
            if self.covariates:  # add covars
                self.beta_variables = [];
                beta_names = []
                for idx, covariate in enumerate(self.covariates):
                    var_name = '_'.join(['beta', str(idx)])
                    beta_var = pm.Normal(var_name, mu=0.0, tau=1)
                    beta_names.append(var_name)
                    self.beta_variables.append(beta_var)
                    mu_ = mu_ + beta_var * covariate

            if self.shifted_response:  # add autoterms
                self.rho_variables = [];
                rho_names = []
                autos = self.shifted_response.copy()

                for idx, autoterm in enumerate(reversed(autos)):
                    var_name = '_'.join(['rho', str(idx + 1)])
                    rho_var = pm.Uniform(var_name, lower=0, upper=1)
                    rho_names.append(var_name)
                    self.rho_variables.append(rho_var)
                    mu_ = mu_ + rho_var * autoterm

            theta_sd = pm.Gamma('theta_sd', alpha=2, beta=2)
            Y = pm.Normal('Y', mu=mu_, tau=theta_sd, observed=self.response)


    def predict_in_sample(self, sample_size=1000, use_median=False):
        self.predicted = super()._predict_in_sample(sample_size=sample_size, use_median=use_median)
        return self.predicted

    def predict(self, new_x=None, steps=1, sample_size=1000):
        '''
        make predictions for future dates (out-of-sample prediction)
        Args:
            steps (int): how many days to forcast
            new_x(list): a list of numpy arrays. The dimension should match number of days
            sample_size (int): sample size of posterior sample
            use_median(boolean): if true, use median as point estimate otherwise mean will be used.
        '''
        if new_x:
            if not self.covariates:
                print('No covariates in original model; thus they should not show up in prediction.')
                print('hence input covariates igored')
                new_x = None
            else:
                shape = new_x[0].shape
                for numpy_array in new_x:
                    if shape != numpy_array.shape:
                        raise ValueError('the dimensions of covariates do not match!')
                else:
                    steps = shape[1]  # 1d sample size, 2d days
                new_x = CarModel.pad_covariates(new_x, steps)

        else:
            if self.covariates:
                raise ValueError('covariates not given. use predict_in_sample() if you want in-sample prediction.')
            else:
                print(f'now predicting future {steps} day(s). Change steps=10 if i.e., 10 terms are desired.')

        if self.autoreg > 0:
            _, y_temp = np.hsplit(self.response, [-self.autoreg])
            last_ys = np.hsplit(y_temp, range(self.autoreg))
            last_ys.pop(0)
        else:
            last_ys = None

        if steps > 1 and new_x:
            x_split = []
            for covariate_ndarray in new_x:
                x_split_one_var = np.hsplit(covariate_ndarray, [i for i in range(steps)])
                x_split_one_var.pop(0)
                x_split.append(x_split_one_var)

            x_by_date = []  # make a list of cov by dates
            for day in range(steps):
                temp_var_one_day = []
                for covariate in x_split:
                    temp_var_one_day.append(covariate[day])
                x_by_date.append(temp_var_one_day)

            del temp_var_one_day
            del x_split

        def predict_one_step(current_x, last_y):
            '''
            the helper function to predit Yt+1
            will be called iteratively to predict multiple days
            '''
            temp_name_for_y = token_hex(2)
            with self.model:
                mean = self.phi.T[:, -1]
                if current_x:  # a list of vars
                    for cov_, beta in zip(current_x, self.beta_variables):
                        mean = mean + cov_.ravel() * beta
                if last_ys is not None:
                    for last_y_, rho in zip(last_ys, self.rho_variables):
                        mean = mean + last_y_.ravel() * rho
                y_temp = pm.Deterministic(temp_name_for_y, mean)
                svs = pm.sample_ppc(self.trace, vars=[y_temp], samples=sample_size)[temp_name_for_y]

                y = np.mean(svs, axis=0)
                y_lower = np.percentile(svs, 2.5, axis=0)
                y_upper = np.percentile(svs, 97.5, axis=0)

            return y, y_lower, y_upper

        if steps == 1:
            return predict_one_step(current_x=new_x, last_y=last_ys)

        elif steps > 1:  # multipe steps
            y_history_rec = []
            idx = 1
            while steps:
                variable_name = '_'.join(['Y', str(idx)])
                if self.covariates:
                    current_x = x_by_date.pop()
                else:
                    current_x = None
                if self.autoreg > 0:  # autoreg in model
                    y_mean, y_lower, y_upper = predict_one_step(current_x, last_ys)
                    last_ys.append(y_mean)
                    last_ys.pop(0)
                else:
                    y_mean, y_lower, y_upper = predict_one_step(current_x, None)

                y_history_rec.append([y_mean, y_lower, y_upper])
                steps -= 1
                idx += 1
            return np.array(y_history_rec)
        else:
            raise ValueError('steps has to be a positive integer!')
