""" Module for the StatTestSettings class. """

from dataclasses import dataclass


@dataclass
class StatTestSettings:
    """Class for settings for statistical tests.

    :param confidence_interval: The confidence interval for the binomial test. Defaults to 0.95.
    :type confidence_interval: float, optional
    :param alpha: The alpha value for the binomial test. Defaults to 0.05.
    :type alpha: float, optional
    """

    def __init__(self, confidence_interval: float = 0.95, alpha: float = 0.05):
        """
        Initialize the StatTestSettings class.

        :param confidence_interval: The confidence interval for the binomial test. Defaults to 0.95.
        :type confidence_interval: float, optional
        :param alpha: The alpha value for the binomial test. Defaults to 0.05.
        :type alpha: float, optional
        """

        self.confidence_interval = confidence_interval
        self.alpha = alpha
