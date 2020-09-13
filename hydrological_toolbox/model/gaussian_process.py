import pymc3 as pm
from model.base_model import Model
"""
loosely speaking, how far do you need length-scale to move (along a particular axis) in input space for the
function values to be automatic relevance come uncorrelated
"""
from pymc3.gp.cov import Constant, WhiteNoise, ExpQuad, Matern52, Matern32, \
    Exponential, Linear, Polynomial, Periodic, Cosine
from pymc3 import Normal, Cauchy
import logging
logger = logging.getLogger(__name__)
supported_kernel_dict = {'Constant': Constant,
                         'WhiteNoise': WhiteNoise,
                         'ExpQuad': ExpQuad,
                         'Matern52': Matern52,
                         'Matern32': Matern32,
                         'Exponential': Exponential,
                         'Linear': Linear,
                         'Polynomial': Polynomial,
                         'Periodic': Periodic,
                         'Cosine': Cosine
                         }

supported_noise_distribution_dict = {'Normal': Normal,
                                     'Cauchy': Cauchy}


class GP(Model):
    """
    Build a spatial temporal model based on Pymc3 framework.
    """

    def __init__(self,
                 hyper_param_length_scale_gamma_a: float = 2,
                 hyper_param_length_scale_gamma_b: float = 1,
                 hyper_param_cov_function_scale_half_cauchy: float = 5,
                 kernel_type: str = 'Matern52',
                 hyper_param_noise_scale_gamma_a: float = 2,
                 hyper_param_noise_scale_gamma_b: float = 0.1,
                 noise_distribution: str = 'Normal'):

        super().__init__()
        self.trace = None
        self.hyper_param_length_scale_gamma_a = hyper_param_length_scale_gamma_a
        self.hyper_param_length_scale_gamma_b = hyper_param_length_scale_gamma_b
        self.hyper_param_cov_function_scale_half_cauchy = hyper_param_cov_function_scale_half_cauchy

        if kernel_type not in supported_kernel_dict:
            raise ValueError(f'{kernel_type} kernel type is not supported,'
                             f' only {supported_kernel_dict.keys()} are supported')
        else:
            self.kernel_type = supported_kernel_dict[kernel_type]

        self.hyper_param_noise_scale_gamma_a = hyper_param_noise_scale_gamma_a
        self.hyper_param_noise_scale_gamma_b = hyper_param_noise_scale_gamma_b

        if noise_distribution not in supported_noise_distribution_dict:
            raise ValueError(f'{supported_noise_distribution_dict} kernel type is not supported,'
                             f' only {supported_noise_distribution_dict.keys()} are supported')
        else:
            self.noise_distribution = supported_noise_distribution_dict[noise_distribution]

        # the prediction based on the given y
        self.y_pred = None
        # the out of sample prediction. Will be populated if user call .predict()
        self.y_pred_out_of_sample = None
        self.new_locations = None

    def fit(self, df, locations_cols, y_col, **kwargs):
        super().fit(df, locations_cols, y_col, **kwargs)
        eval_dim = self.locations.shape[1]
        with pm.Model() as self.model:
            gamma_rv_for_length_scale = pm.Gamma(
                name='length_scale',
                alpha=self.hyper_param_length_scale_gamma_a,
                beta=self.hyper_param_length_scale_gamma_b)

            half_cauchy_rv = pm.HalfCauchy(
                name='scale_for_kernel',
                beta=self.hyper_param_cov_function_scale_half_cauchy)

            cov_kernel_func = half_cauchy_rv ** 2 * self.kernel_type(eval_dim, gamma_rv_for_length_scale)
            self.gp = pm.gp.Latent(cov_func=cov_kernel_func)

            gaussian_process_mean = self.gp.prior('f', X=self.locations)

            sigma_for_normal = pm.Gamma(
                name='noise_scale',
                alpha=self.hyper_param_noise_scale_gamma_a,
                beta=self.hyper_param_length_scale_gamma_b)

            y = pm.Normal(name='y',
                          mu=gaussian_process_mean,
                          sigma=sigma_for_normal,
                          observed=self.y)

            self.trace = pm.sample(tune=2000,
                                   chains=4,
                                   cores=12,
                                   **kwargs)

    def predict(self, new_data=None, location_cols=None, **kwargs):

        if not new_data:
            logger.critical("no new location is given hence we are making prediction on original locations.")
            return self.trace['f'].mean(axis=0)

        self.new_locations = new_data[location_cols].values
        with self.model:
            predicted_value = self.gp.conditional('y_pred', self.new_locations)
            pred_samples = pm.sample_posterior_predictive(self.trace,
                                                          vars=[predicted_value],
                                                          samples=1000)
        return pred_samples

    def summarize(self):
        """
        a summary for all the involved parameters with point estimate and
        credible interval
        """
        return pm.summary(self.trace).round(2)



if __name__ == '__main__':
    import numpy as np
    import pickle
    import pandas as pd

    n = 10  # The number of data points
    X = np.linspace(0, 10, n)[:, None]  # The inputs to the GP, they must be arranged as a column vector

    # Define the true covariance function and its parameters
    l_true = 1.0
    e_true = 3.0
    cov_func = e_true ** 2 * pm.gp.cov.Matern52(1, l_true)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    f_true = np.random.multivariate_normal(mean_func(X).eval(),
                                           cov_func(X).eval() + 1e-8 * np.eye(n), 1).flatten()

    # The observed data is the latent function plus a small amount of T distributed noise
    # The standard deviation of the noise is `sigma`, and the degrees of freedom is `nu`
    s_true = 1.0
    n_true = 3.0
    y = f_true + s_true * np.random.normal(loc=0, scale=n_true, size=n)

    gaussian_process_model = GP()
    df = pd.DataFrame([X.flatten(), y]).T
    df.columns = ['x', 'y']
    gaussian_process_model.fit(df=df, locations_cols=['x'], y_col=['y'])
    with open('temp.pkl', 'wb') as h:
        pickle.dump(gaussian_process_model, h, protocol=pickle.HIGHEST_PROTOCOL)

# v = GP()
# import numpy as np
#
# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = np.random.standard_normal(100)
# from pandas import DataFrame
# training = DataFrame([x, y, z])
# training = training.T
# training.columns = ['x', 'y', 'z']
#
# v.fit(training, ['x', 'y'], ['z'])
# exit()
#


#
#
#
#
# @property
# def valid_kernel(self):
#     source = """https://docs.pymc.io/api/gp/cov.html#"""
#     if self._kernel not in supported_kernel_dict:
#         raise KeyError(f'only support the following kernels {supported_kernel_dict.keys()}')
#     else:
#         kernel_func = supported_kernel_dict[self._kernel]
#
#     if not isinstance(self._kernel_params, dict):
#         raise TypeError(f'params arg needs to be a dictionary with parameter name and value for {self._kernel}')
#     for param in self._kernel_params:
#         if param not in supported_kernel_parameters[self._kernel]:
#             raise ValueError(f'{param} is not support for {self._kernel}. '
#                              f'Go to {source} to checkout support parameters in each type of kernel')
#     else:
#         valid_kernel = kernel_func(self._kernel_params)
#     return valid_kernel
#
#
#
#
# def fit(self, sampling_size = 5000, traceplot_name = None, fast_sampling = False):
#     '''
#     Args:
#         sampling_size (int): the length of markov chain
#         create_traceplot (boolean): Whether or not generate the traceplot.
#     '''
#     self.model = pm.Model()
#     with self.model:
#         rho = pm.Exponential('rho', 1/5, shape = 3)
#         tau = pm.Exponential('tau', 1/3)
#
#         cov_func = pm.gp.cov.Matern52(3, ls = rho)
#         self.gp = pm.gp.Marginal(cov_func = cov_func)
#
#         sigma = pm.HalfNormal('sigma', sd = 3)
#         y_ = self.gp.marginal_likelihood('y',
#                                     X = self.X_train,
#                                     y = np.log(self.y_train),
#                                     noise = sigma)
#
#     if fast_sampling:
#         with self.model:
#             inference = pm.ADVI()
#             approx = pm.fit(n = 50000, method=inference) #until converge
#             self.trace = approx.sample(draws = sampling_size)
#
#     else:
#         with self.model:
#             start = pm.find_MAP()
#             self.trace = pm.sample(sampling_size, nchains = 1)
#
#     if traceplot_name:
#         fig, axs = plt.subplots(3, 2) # 2 RVs
#         pm.traceplot(self.trace, varnames = ['rho', 'sigma', 'tau'], ax = axs)
#         fig.savefig(traceplot_name)
#         fig_path = os.path.join(os.getcwd(), traceplot_name)
#         print(f'the traceplot has been saved to {fig_path}')
#
# def predict(self, new_df = None, sample_size = 1000):
#     '''
#     Args:
#         new_data_frame (pandas dataframe): the dataframe of new locations. Users can also include the truth value of Y.
#         Note that MSE cannot be computed if truth is not provided.
#     '''
#     if new_df:
#         try:
#             self.X_test = coordinates_converter(new_df)
#             self.y_test = new_df[self.response_var]
#             self.test_loc_cache = new_df[['LATITUDE','LONGITUDE']]
#         except:
#             raise ValueError('The new dataframe should contain LATITUDE, LONGITUDE and the variable column, e.g., PRCP')
#
#     with self.model:
#         y_pred = self.gp.conditional("y_pred", self.X_test)
#         self.simulated_values = pm.sample_ppc(self.trace, vars=[y_pred], samples= sample_size)
#         self.predictions = np.exp(np.median(self.simulated_values['y_pred'], axis = 0))
#
#     l1_loss = np.mean(np.abs(self.predictions - self.y_test))
#     l2_loss = np.mean(np.square(self.predictions - self.y_test))
#     self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}
#
#     output_df = self.test_loc_cache.copy()
#     output_df['PRED'] = self.predictions
#
#     return self.predictions
