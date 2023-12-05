""" class UnivariateGaussian: parametes of univariate Gaussian distributions"""
import dataclasses
import typing

import numpy as np

@dataclasses.dataclass
class UnivariateGaussian:
    """
    Represents a univariate Gaussian distribution.

    Attributes:
        mean (Union[float, np.ndarray]): The mean of the distribution. Can be a single value or a (1, n) vector of means.
        var (Union[float, np.ndarray]): The variance of the distribution. Can be a single value or a (1, n) vector of variances.
    """
    mean: typing.Union[float, np.ndarray]
    var: typing.Union[float, np.ndarray]
