""" functions for calculating calibration of predictions """""
import typing

import numpy as np
import scipy.stats

from .gaussian import UnivariateGaussian


def calc_calibration(calc_percentiles: np.ndarray, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                     output_data: np.ndarray) -> np.ndarray:
    """ Calculate the calibration of predictions based on the given percentiles and prediction type.

    :param calc_percentiles: An array of percentiles used for calibration.
    :type calc_percentiles: numpy.ndarray
    :param predictions: The predictions to be calibrated.
    :param prediction_type: The type of prediction used for calibration, 
    should be one of the values from the :class:`PredictionType` enumeration.
    :type prediction_type: PredictionType
    :param output_data: The output data used for calibration.
    :returns: The observed CDF (Cumulative Distribution Function) after calibration as a numpy.ndarray.
    :rtype: numpy.ndarray
    """
    if isinstance(predictions, np.ndarray):
        observed_cdf = calc_calibration_sampled_prediction(
            calc_percentiles=calc_percentiles, predictions=predictions, output_data=output_data)
    elif isinstance(predictions, UnivariateGaussian):
        observed_cdf = calc_calibration_gaussian_prediction(
            calc_percentiles=calc_percentiles, predictions=predictions, output_data=output_data)
    else:
        raise ValueError('invalid prediction type')
    return observed_cdf


def calc_calibration_sampled_prediction(calc_percentiles: np.ndarray,
                                        predictions,
                                        output_data: np.ndarray) -> np.ndarray:
    """
    Calculates the calibration curve of a prediction model using percentiles.

    :param calc_percentiles: The percentile to be used for the calculation.
    :type calc_percentiles: int
    :param predictions: The predictions made by the model.
    :type predictions: numpy.ndarray
    :param output_data: The actual outputs for the inputs used in the model.
    :type output_data: numpy.ndarray
    :return: The observed CDF (cumulative distribution function) of the model.
    :rtype: numpy.ndarray
    """
    percentiles = np.percentile(predictions, calc_percentiles, axis=0)

    output_data = np.swapaxes(output_data, 0, 1)
    comp_cdf = np.less(output_data, percentiles)
    observed_cdf = 1 / output_data.size * np.sum(comp_cdf, axis=1)

    return observed_cdf


def calc_calibration_gaussian_prediction(calc_percentiles: np.ndarray,
                                         predictions: UnivariateGaussian,
                                         output_data: np.ndarray) -> np.ndarray:
    """
    Calculates the calibration curve of a prediction model using percentiles.

    :param calc_percentiles: The percentile to be used for the calculation.
    :type calc_percentiles: int
    :param predictions: The predictions made by the model.
    :type predictions: numpy.ndarray
    :param output_data: The actual outputs for the inputs used in the model.
    :type output_data: numpy.ndarray
    :return: The observed CDF (cumulative distribution function) of the model.
    :rtype: numpy.ndarray
    """

    if not isinstance(predictions, UnivariateGaussian):
        raise ValueError('invalid prediction type')

    std = np.sqrt(predictions.var)
    num_predictions = predictions.mean.shape[0]
    num_percentiles = calc_percentiles.shape[0]
    calc_percentiles = calc_percentiles / 100.  # rescale to numbers between 0 and 1
    # calculate gaussian percentiles
    gaussian_percentile = np.empty((num_percentiles, num_predictions))
    for i in range(0, num_predictions):
        gaussian_percentile[:, i] = scipy.stats.norm.ppf(
            calc_percentiles, loc=predictions.mean[i], scale=std[i])

    output_data = np.swapaxes(output_data, 0, 1)
    comp_cdf = np.less(output_data, gaussian_percentile)
    observed_cdf = 1 / output_data.size * np.sum(comp_cdf, axis=1)

    return observed_cdf
