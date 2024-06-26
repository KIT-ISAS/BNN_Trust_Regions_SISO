""" distance functions """


import typing

import numpy as np


from .gaussian import UnivariateGaussian


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
        var_prediction = prediction.var
    else:
        mean_prediction = np.mean(prediction, axis=0)
        var_prediction = np.var(prediction, axis=0)
    output_data = np.squeeze(output_data)

    inverse_var = 1 / var_prediction
    return np.square(output_data - mean_prediction) * inverse_var
