""" class CandidateRegion """


from dataclasses import dataclass
import typing

import numpy as np

from .gaussian import UnivariateGaussian


@dataclass
class CandidateRegion:
    x_min: float
    x_max: float

    predictions_in_region: typing.Union[np.ndarray, UnivariateGaussian]
    outputs_in_region: np.ndarray

    # TODO add instances for statistical test results

    # anees: float
    # anees_crit_bounds: np.ndarray

    # confidence_level: float
    # proportion_inside: float

    def __init__(self, predictions_in_region: typing.Union[np.ndarray, UnivariateGaussian],
                 outputs_in_region: np.ndarray,
                 x_min: float,
                 x_max: float,):

        self.x_min = x_min
        self.x_max = x_max

        self.predictions_in_region = predictions_in_region
        self.outputs_in_region = outputs_in_region
