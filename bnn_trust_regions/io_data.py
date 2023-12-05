""" class for input and output data"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IOData:
    """
    Represents input-output data for a model.

    :param input: The input data for the model.
    :type input: np.ndarray
    :param output: The output data for the model.
    :type output: np.ndarray
    """
    input: np.ndarray
    output: np.ndarray
