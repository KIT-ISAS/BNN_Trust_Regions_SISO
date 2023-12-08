
import typing

import numpy as np
from scipy import stats

from ..gaussian import UnivariateGaussian


def ci_to_quantile_level(confidence_interval):
    """ Convert confidence interval to quantile level.
    :param confidence_interval: The confidence interval of the predictions.
    :type confidence_interval: float
    :return: The quantile level of the predictions.
    :rtype: np.ndarray

    example:
    >>> ci_to_quantile_level(0.95)
    array([0.025, 0.975])
    """
    lower_quantile_level = (1 - confidence_interval) / 2
    return np.array([lower_quantile_level, 1 - lower_quantile_level])  # quantile_level


def calc_mean_and_quantiles(predictions: typing.Union[np.ndarray, UnivariateGaussian], confidence_interval: float):
    """Calculate the mean and quantiles of the predictions.
    :param predictions: The predictions.
    :type predictions: Union[np.ndarray, UnivariateGaussian]
    :param confidence_interval: The confidence interval of the predictions.
    :type confidence_interval: float
    :return: The mean and quantiles of the predictions.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    quantile_level = ci_to_quantile_level(confidence_interval)

    if isinstance(predictions, UnivariateGaussian):
        mean_predictions = predictions.mean
        var_predictions = predictions.var
        # calculate confidence interval of gaussian predictions by using the inverse cdf
        # of the normal distribution
        quantiles = stats.norm.ppf(quantile_level, loc=mean_predictions,
                                   scale=np.sqrt(var_predictions))

    if isinstance(predictions, np.ndarray):
        num_samples_per_prediction, _ = predictions.shape

        if num_samples_per_prediction == 1:  # only one sample per prediction
            mean_predictions = predictions[0, :]
            quantiles = None
        else:
            mean_predictions = predictions.mean(axis=0)
            quantiles = np.quantile(predictions, quantile_level, axis=0)

    return mean_predictions, quantiles
