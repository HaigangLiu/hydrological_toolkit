import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import logging

logger = logging.getLogger(__name__)


def plot(spatial_model):
    """
    takes a spatial model, and this model must have location, residuals
    """
    pass


def plot_car_model(residuals, locations, kind):
    """
    this is a helper function to plot the residuals from the car mode prediction,
    which can automatically analyze the residual based on space dimension or time dimension.
    """

    plot_kinds = ['histogram', 'time series', 'acf', 'pacf']

    if kind not in plot_kinds:
        raise ValueError(f'kind {kind} is not supported; only support {plot_kinds}')

    residuals = pd.DataFrame(residuals, index=locations)
    regional_resid = residuals.groupby(level=0).mean()  # column is day, row is river basin

    if kind.lower() in 'histogram':
        figure, axes = plt.subplots(1, 1)
        _ = regional_resid.T.hist(ax=axes, sharex=True, sharey=True)
        plt.tight_layout()

    elif kind.lower() in 'time series':
        figure, axes = plt.subplots(1, 1)
        _ = regional_resid.T.plot(ax=axes)

    elif kind.lower() in ['acf', 'pacf']:
        if kind.lower() == 'acf':
            per_loc = regional_resid.apply(acf, nlags=10, axis=1)  # pd.Series object
        else:
            per_loc = regional_resid.apply(pacf, nlags=10, axis=1)  # pd.Series object
        graph_height = 2 * len(per_loc)
        figure, axes = plt.subplots(1, 1, figsize=[10, graph_height])
        per_loc = pd.DataFrame(per_loc.values.tolist(), index=per_loc.index)
        per_loc.T.plot(kind='bar', subplots=True, ax=axes)
        plt.tight_layout()


def summarize_input(size, days, covars):
    logger.info('-' * 40)
    logger.info('BASIC INFO FROM INPUT DATA')
    logger.info('-' * 40)
    logger.info(f'SAMPLE SIZE:{size}')
    logger.info(f'TIME SPAN: {days}')
    logger.info(f'COVAR: {covars}')
