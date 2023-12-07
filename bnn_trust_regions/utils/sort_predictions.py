import typing

import numpy as np

from ..gaussian import UnivariateGaussian


def sort_predictions(predictions: typing.Union[np.ndarray, UnivariateGaussian], idx: np.ndarray):
    """
    The function `sort_predictions` takes in predictions and an index array and sorts the predictions
    based on the index.

    :param predictions: The `predictions` parameter can be either a numpy array or an instance of the
    `UnivariateGaussian` class. It represents the predicted values or probability distribution of a
    variable
    :type predictions: typing.Union[np.ndarray, UnivariateGaussian]
    :param idx: The `idx` parameter is a numpy array that specifies the indices of the elements to be
    selected from the `predictions` array
    :type idx: np.ndarray
    :return: either the modified predictions object (if it is an instance of UnivariateGaussian) or the
    modified predictions array (if it is a numpy array).
    """

    if isinstance(predictions, UnivariateGaussian):
        predictions.mean = predictions.mean[idx]
        predictions.var = predictions.var[idx]
        return predictions
    if isinstance(predictions, np.ndarray):
        return predictions[:, idx]
    raise ValueError(
        "predictions must be either a numpy array or an instance of UnivariateGaussian")
