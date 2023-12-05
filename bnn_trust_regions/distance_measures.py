""" distance functions """


import typing

import numpy as np


from gaussian import UnivariateGaussian


def mean_square_error(x_pred, x_observed):
    """
    Calculates the mean squared error between predicted and observed values.

    :param x_pred: A numpy array of predicted values.
    :type x_pred: numpy.ndarray
    :param x_observed: A numpy array of observed values.
    :type x_observed: numpy.ndarray
    :return: The mean squared error between the predicted and observed values.
    :rtype: float
    """
    if isinstance(x_pred, UnivariateGaussian):
        x_pred = x_pred.mean
    return np.square(x_pred - x_observed.T).mean(axis=None)


def squared_mahalanobis_distance(prediction: typing.Union[np.ndarray, UnivariateGaussian], output_data: np.ndarray,):
    """
    The function calculates the squared Mahalanobis distance between a prediction and output data.

    :param prediction: The `prediction` parameter can be either a numpy array or an instance of the
    `Gaussian` class. If it is a numpy array, it represents the predicted values. If it is
    an instance of `Gaussian`, it contains the mean and covariance of
    :type prediction: typing.Union[np.ndarray, Gaussian]
    :param output_data: The `output_data` parameter is a numpy array that contains the actual output
    values
    :type output_data: np.ndarray
    :return: the squared Mahalanobis distance between the prediction and the output data.
    """
    if isinstance(prediction, UnivariateGaussian):
        mean_prediction = prediction.mean
        var_prediction = prediction.cov
    else:
        mean_prediction = np.mean(prediction, axis=0)
        var_prediction = np.var(prediction, axis=0)
    output_data = np.squeeze(output_data)

    inverse_var = 1 / var_prediction
    return np.square(output_data - mean_prediction) * inverse_var
