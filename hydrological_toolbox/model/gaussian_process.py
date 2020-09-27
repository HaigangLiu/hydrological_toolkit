import logging

import pymc3 as pm
from pymc3 import Cauchy, Normal, Laplace, StudentT, InverseGamma, Gamma, HalfNormal
from pymc3.gp.cov import Constant, WhiteNoise, ExpQuad, Matern52, Matern32, \
    Exponential, Linear, Polynomial, Periodic, Cosine

logger = logging.getLogger(__name__)


class GP:
    """
    Build a spatial model based on Gaussian process.
    Interpretation of several parameters (self.kernel_type and self.kernel_parameter)
    1. kernel choice and ls parameter
    determines the spatial structure, and ls (length scale determines the strength of the parameter)

    In other words, how far do you need length-scale to move (along a particular axis) in input space for the
    function values to be automatic relevance come uncorrelated

    2. Error type:
    what distribution are the residuals following. the default is normal distribution,
    but for heavy tail distributions, student t or laplace could be used as well

    3. The scale parameter for the kernel
    the multiplier in front of the kernel
    """
    positive_continuous_distributions = {'Cauchy': Cauchy,
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

        self.error_distribution = None
        self.error_parameter = None

        # the multiplier before the cov(kernel) function
        self.scale_distribution_params = None
        self.scale_distribution_for_kernel = None

    def set_error_type(self, error_type=None, **kwargs):
        """
        use kwargs to allow versatile input, because the options(parameters) are different for different distributions.
        """
        with self.model:
            if error_type not in self.positive_continuous_distributions:
                raise KeyError('this error type is not supported')
            self.error_parameter = kwargs
            self.error_distribution = self.positive_continuous_distributions[error_type]
            return self

    def set_kernel_type(self, kernel_type=None, **kwargs):
        """
        use kwargs to allow versatile input, because the options are different for different kernel types
        """
        with self.model:
            self.kernel_type, self.kernel_parameter = self.gaussian_kernels[kernel_type], kwargs
            return self

    def set_scale_parameter_for_kernel(self, scale_distribution, **kwargs):
        """
        use kwargs to allow versatile input, because the options are different for different kernel types
        """
        with self.model:
            if scale_distribution not in self.positive_continuous_distributions:
                raise KeyError('this distribution is not supported for scale parameter')
            if kwargs and 'name' not in kwargs:
                kwargs.update({'name': 'scale parameter for gp kernel'})

            self.scale_distribution_params = kwargs
            self.scale_distribution_for_kernel = self.positive_continuous_distributions[scale_distribution]
            return self

    def sample(self, y, locations, X=None, approximation=False, **kwargs):
        self.X = X
        self.y = y
        self.locations = locations

        with self.model:
            if self.kernel_type is None:
                self.kernel_type = Matern52
                self.kernel_parameter = {'ls': pm.InverseGamma(name='rho: spatial correlation', alpha=1, beta=1),
                                         'input_dim': self.locations.shape[1]}

            gp_kernel = self.kernel_type(**self.kernel_parameter)

            if self.scale_distribution_for_kernel is None:
                self.scale_distribution_for_kernel = InverseGamma
                self.scale_distribution_params = {'name': 'kernel scale multiplier', 'alpha': 1, 'beta': 1}

            scale_for_kernel = pm.math.sqr(self.scale_distribution_for_kernel(**self.scale_distribution_params))

            if self.error_distribution is None:
                self.error_distribution = Normal
                self.error_parameter = {'sigma': InverseGamma.dist(alpha=1, beta=1)}

            cov_kernel_func = scale_for_kernel * gp_kernel

            self.gp = pm.gp.Latent(cov_func=cov_kernel_func)
            gaussian_process_mean = self.gp.prior('f', X=self.locations)
            y_ = self.error_distribution(name='y',
                                         mu=gaussian_process_mean,
                                         observed=self.y,
                                         **self.error_parameter)
            if approximation:
                self.trace = pm.fit(method='advi', n=100_00).sample()
            else:
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

