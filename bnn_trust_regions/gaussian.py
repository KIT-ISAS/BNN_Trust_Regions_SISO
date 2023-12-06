""" class UnivariateGaussian: parametes of univariate Gaussian distributions"""
import dataclasses
import typing
from tqdm import tqdm

import numpy as np
import scipy.stats


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

    def get_gaussian_quantiles(self, quantiles_lower_tail_probability: np.ndarray):
        """
        Calculates Gaussian quantiles based on the lower tail probability.

        :param quantiles_lower_tail_probability: Array of lower tail probabilities for which to calculate Gaussian quantiles.
        :type quantiles_lower_tail_probability: numpy.ndarray
        :return: Array of Gaussian quantile positions.
        :rtype: numpy.ndarray
        """
        num_elements = len(self.mean)
        num_quantiles = len(quantiles_lower_tail_probability)
        std = np.sqrt(self.var)
        quantile_position = np.empty((num_quantiles, num_elements))
        for i in tqdm(range(0, num_elements), desc="Progress", disable=True):
            quantile_position[:, i] = scipy.stats.norm.ppf(
                quantiles_lower_tail_probability, loc=self.mean[i], scale=std[i])
        return quantile_position
