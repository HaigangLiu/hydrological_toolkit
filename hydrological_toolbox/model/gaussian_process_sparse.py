import logging

import pymc3 as pm
from pymc3 import Cauchy, Normal, Laplace, StudentT, InverseGamma, Gamma, HalfNormal, HalfCauchy
from pymc3.gp.cov import Constant, WhiteNoise, ExpQuad, Matern52, Matern32, \
    Exponential, Linear, Polynomial, Periodic, Cosine

logger = logging.getLogger(__name__)


class GP:
    """Build a spatial model based on Gaussian process.

    Highlights:
    1. used approximations to speed up the calculation.
    These GP approximations don’t form the full covariance matrix over all n training inputs.
    Instead they rely on m << n inducing points, which are "strategically" placed throughout the domain.
    see Quinonero, Candela and Rasmussen (2006) for more about this approximation

    2. gaussian process is used to model the spatial structure. The available structures can be found in the
    `gaussian_kernels` variable.

    3. Three most important parameters from this model are
    * (rho): spatial correlation
        interpretation: how far we need to go before the observations get independent
    * (sigma): noise scale
        interpretation: the scale of white noise parameter sigma. affects the spread of final distribution
    * (eta): kernel scale multiplier
        interpretation: the multiplier before the gp kernel function
    """
    positive_continuous_distributions = {'Cauchy': Cauchy,
                                         'HalfCauchy': HalfCauchy,
                                         'Normal': Normal,
                                         'HalfNormal': HalfNormal,
                                         'Laplace': Laplace,
                                         'Gamma': Gamma,
                                         'InverseGamma': InverseGamma,
                                         'StudentT': StudentT}

    gaussian_kernels = {'Constant': Constant,
                        'WhiteNoise': WhiteNoise,
                        'ExpQuad': ExpQuad,
                        'Matern52': Matern52,
                        'Matern32': Matern32,
                        'Exponential': Exponential,
                        'Linear': Linear,
                        'Polynomial': Polynomial,
                        'Periodic': Periodic,
                        'Cosine': Cosine}

    def __init__(self):
        # for model
        self.model = pm.Model()
        self.y = None
        self.X = None
        self.locations = None

        # for prediction
        self.X_new = None
        self.locations_new = None
        self.y_pred = None

        # specifically for GP models
        self.trace = None
        self.gp = None

        # parameterization
        self.kernel_type = None
        self.kernel_parameter = None

        self.error_scale_distribution = None
        self.error_scale_parameter = None

        # the multiplier before the cov(kernel) function
        self.scale_distribution_params = None
        self.scale_distribution_for_kernel = None

    def set_error_scale(self, error_type=None, **kwargs):
        """
        the hyperparameter for the normal error. the variable is supposed to be from a non-negative distribution
        """
        with self.model:
            if error_type not in self.positive_continuous_distributions:
                raise KeyError('this error type is not supported')
            self.error_scale_parameter = kwargs
            self.error_scale_distribution = self.positive_continuous_distributions[error_type]
            return self

    def set_gp_kernel_type(self, kernel_type=None, **kwargs):
        """
        use kwargs to allow versatile input, because the options are different for different kernel types
        """
        with self.model:
            self.kernel_type, self.kernel_parameter = self.gaussian_kernels[kernel_type], kwargs
            return self

    def set_scale_param_for_kernel(self, scale_distribution, **kwargs):
        """
        use kwargs to allow versatile input, because the options are different for different kernel types
        """
        with self.model:
            if scale_distribution not in self.positive_continuous_distributions:
                raise KeyError('this distribution is not supported for scale parameter')
            if kwargs and 'name' not in kwargs:
                kwargs.update({'name': '(eta): scale param for kernel'})
            self.scale_distribution_params = kwargs
            self.scale_distribution_for_kernel = self.positive_continuous_distributions[scale_distribution]
            return self

    def sample(self, y, locations, X=None, **kwargs):
        self.X = X
        self.y = y
        self.locations = locations

        param_dicts_and_new_names = zip((self.error_scale_parameter,
                                         self.scale_distribution_params),
                                        ('(sigma): noise scale',
                                         '(eta): kernel scale multiplier'))

        for param_dicts, new_name in param_dicts_and_new_names:
            if 'name' not in param_dicts:
                param_dicts.update({'name': new_name})

        with self.model:

            if self.kernel_type is None:
                self.kernel_type = Matern52
                self.kernel_parameter = {'ls': pm.InverseGamma(name='(rho): spatial correlation', alpha=1, beta=1),
                                         'input_dim': self.locations.shape[1]}

            gp_kernel = self.kernel_type(**self.kernel_parameter)

            if self.scale_distribution_for_kernel is None:
                self.scale_distribution_for_kernel = InverseGamma
                self.scale_distribution_params = {'alpha': 1, 'beta': 1}

            scale_for_kernel = pm.math.sqr(self.scale_distribution_for_kernel(**self.scale_distribution_params))

            if self.error_scale_distribution is None:
                self.error_scale_distribution = HalfCauchy
                self.error_scale_parameter = {'beta': 5}

            self.gp = pm.gp.MarginalSparse(cov_func=scale_for_kernel * gp_kernel, approx="FITC")
            inducing_points = pm.gp.util.kmeans_inducing_points(20, self.locations)
            error_variable = self.error_scale_distribution(**self.error_scale_parameter)

            y_ = self.gp.marginal_likelihood("y",
                                             X=self.locations,
                                             Xu=inducing_points,
                                             y=self.y,
                                             noise=error_variable)
            self.trace = pm.sample(**kwargs)

    def predict(self, locations_new, X_new=None, **kwargs):
        if self.model is None:
            raise Exception('the model has not been fitted yet; consider calling .fit() method first')

        self.locations_new = locations_new
        self.X_new = X_new

        with self.model:
            # add the GP conditional to the model, given the new X values
            predicted_value = self.gp.conditional('y_pred', self.locations_new)
            # Sample from the GP conditional distribution
            pred_samples = pm.sample_posterior_predictive(self.trace, **kwargs)
        return pred_samples

    def summarize(self):
        """
        a summary for all the involved parameters with point estimate and credible interval
        """
        clean_names = []
        for name in self.trace.varnames:
            if name.endswith('__'):
                # any type of transformed variables
                continue
            elif name.endswith('rotated_'):
                continue
            elif name == 'f':  # name for gaussian process
                continue
            else:
                clean_names.append(name)
        return pm.summary(self.trace, var_names=clean_names, round_to=4)
