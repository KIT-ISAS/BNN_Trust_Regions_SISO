""" class for input and output data"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IOData:
    input: np.ndarray
    output: np.ndarray
