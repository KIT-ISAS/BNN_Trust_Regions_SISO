""" This module contains classes for statistical tests for evaluating the quality of uncertainty estimates."""
import abc
from dataclasses import dataclass
import typing

import numpy as np
import scipy.stats

from .distance_measures import squared_mahalanobis_distance
from .gaussian import UnivariateGaussian


# @dataclass
class TestResult:
    """ Base class for test results
    :param accept_stat: The `accept_stat` parameter is a boolean that represents whether the null hypothesis is rejected or not.
    :type accept_stat: bool
    :param pvalue: The `pvalue` parameter is a float that represents the p-value of the test.
    :type pvalue: float
    """
    accept_stat: bool
    pvalue: float
    over_estimation_flag: bool = None

    def __init__(self, accept_stat: bool, pvalue: float):
        self.accept_stat = accept_stat
        self.pvalue = pvalue

    @property
    def over_estimation(self):
        """
        The function returns whether the null hypothesis is rejected because of overestimated uncertainty.
        :return: `True` if the null hypothesis is rejected because of overestimated uncertainty, `False` otherwise.
        """
        if self.over_estimation_flag is None:
            raise ValueError(
                "The over_estimation_flag is not set. Please set the over_estimation_flag.")
        return self.over_estimation_flag and not self.accept_stat

    @property
    def under_estimation(self):
        """
        The function returns whether the null hypothesis is rejected because of underestimated uncertainty.
        :return: `True` if the null hypothesis is rejected because of underestimated uncertainty, `False` otherwise.
        """
        if self.over_estimation_flag is None:
            raise ValueError(
                "The over_estimation_flag is not set. Please set the over_estimation_flag.")
        return not self.over_estimation_flag and not self.accept_stat

    @abc.abstractmethod
    def _is_over_estimating(self):
        """
        The function checks if the uncertainty is overestimated.
        """
        raise NotImplementedError


@dataclass
class BinomTestResult(TestResult):
    """ test result for binomial test
    :param prop_inside: The `prop_inside` parameter is a float that represents the proportion of
    predictions inside the confidence interval.
    :type prop_inside: float
    :param pvalue: The `pvalue` parameter is a float that represents the p-value of the binomial test.
    :type pvalue: float
    :param prop_ci: The `prop_ci` parameter is a numpy array that represents the confidence interval
    for the proportion of predictions inside the confidence interval. It has a shape of `(2,)` and
    contains the lower and upper bound for the confidence interval.
    :type prop_ci: np.ndarray
    :param accept_stat: The `accept_stat` parameter is a boolean that represents whether the null
    hypothesis is rejected or not.
    :type accept_stat: bool
    """

    # Other instance variables
    prop_inside: float
    prop_ci: np.ndarray  # [0] is lower bound, [1] is upper bound

    # Class variable (static variable)
    tested_proportion: float = None

    def __init__(self, prop_inside: float, pvalue: float, prop_ci: np.ndarray, accept_stat: bool):
        super().__init__(accept_stat=accept_stat, pvalue=pvalue)
        self.prop_inside = prop_inside
        self.prop_ci = prop_ci
        self._is_over_estimating()

    def _is_over_estimating(self):
        """
        check if uncertainty is overestimated based on the proportion confidence interval
        """
        # is the proportion to be tested lower than the lower bound of the proportion confidence interval?
        self.over_estimation_flag = self.prop_ci_low() > self.tested_proportion

    def prop_ci_low(self):
        """
        The function returns the lower bound of the confidence interval for a proportion.
        :return: The lower bound of the confidence interval for the proportion.
        """
        return self.prop_ci[0]

    def prop_ci_high(self):
        """
        The function returns the upper bound of the confidence interval for a proportion.
        :return: the second element of the `prop_ci` list.
        """
        return self.prop_ci[1]


class AneesTestResult(TestResult):
    """ test result for ANEES test
    :param anees: The `anees` parameter is a float that represents the ANEES (Average Normalized
    Estimation Error Squared) value.
    :type anees: float
    :param anees_crit_bounds: The `anees_crit_bounds` parameter is a numpy array that represents the critical bounds for ANEES.
    It has a shape of `(2,)` and contains the lower and upper bound for ANEES.
    :type anees_crit_bounds: np.ndarray
    :param pvalue: The `pvalue` parameter is a float that represents the p-value of the ANEES test.
    :type pvalue: float
    :param accept_stat: The `accept_stat` parameter is a boolean that represents whether the null hypothesis is rejected or not.
    :type accept_stat: bool
    :param nees_is_chi2: The `nees_is_chi2` parameter is a boolean that represents whether 
    the normalized estimation error squared (NEES) is chi2 distributed.
    :type nees_is_chi2: bool
    """
    anees: float
    anees_crit_bounds: np.ndarray
    nees_is_chi2: bool

    def __init__(self, anees: float, anees_crit_bounds: np.ndarray, pvalue: float, accept_stat: bool, nees_is_chi2: bool):
        super().__init__(accept_stat=accept_stat, pvalue=pvalue)
        self.anees = anees
        self.anees_crit_bounds = anees_crit_bounds
        self.nees_is_chi2 = nees_is_chi2
        self._is_over_estimating()

    def anees_crit_bound_low(self):
        """
        The function returns the lower bound of the ANEES (Average Normalized Estimation Error Squared)
        critical bounds.
        :return: the first element of the `anees_crit_bounds` list.
        """
        return self.anees_crit_bounds[0]

    def anees_crit_bound_high(self):
        """
        The function returns the upper bound of the critical value for ANEES.
        :return: the second element of the list `self.anees_crit_bounds`.
        """
        return self.anees_crit_bounds[1]

    def _is_over_estimating(self):
        """
        check if uncertainty is overestimated based on the ANEES critical bounds
        """
        # is the ANEES value lower than the lower bound of the ANEES critical bounds?
        self.over_estimation_flag = self.anees < self.anees_crit_bound_low()


@dataclass
class StatTest:
    """ Base class for statistical tests

    :param alpha: The `alpha` parameter is a float that represents the significance level of the test.
    :type alpha: float
    """
    alpha: float


@dataclass
class BinomialTest(StatTest):
    """ Binomial test

    :param confidence_interval: The `confidence_interval` parameter is a float that represents the
    confidence interval to be tested by the binomial test. Defaults to 0.95.
    :type confidence_interval: float, optional 
    :param alpha: The `alpha` parameter is a float that represents the significance level of the test.
    """
    confidence_interval: float  # e.g. 0.95 for testing of 95% confidence interval
    # e.g. [0.025 0.975] for testing of 95% confidence interval
    _two_tailed_ci_levels: np.ndarray

    # alpha: float  # e.g. 0.05 for significance level of 5%

    def __init__(self, confidence_interval: float, alpha: float):
        # constructor of parent class
        super().__init__(alpha=alpha)
        self.confidence_interval = confidence_interval
        # self.alpha = alpha
        BinomTestResult.tested_proportion = confidence_interval

        two_tailed = (1 - self.confidence_interval) / 2
        self._two_tailed_ci_levels = np.array([two_tailed, 1-two_tailed])

    def calc_two_sided_binomial_test(self, prediction: typing.Union[np.ndarray, UnivariateGaussian], output_data: np.ndarray):
        """
        The function calculates a two-sided binomial test to determine if the proportion of successes in
        a prediction is significantly different from a null hypothesis proportion.

        :param prediction: The `prediction` parameter can be either a numpy array or an instance of the
        `UnivariateGaussian` class. It represents the predicted values or the distribution of predicted
        values
        :type prediction: typing.Union[np.ndarray, UnivariateGaussian]
        :param output_data: The `output_data` parameter is a numpy array that contains the observed data
        or outcomes. It represents the results of a binary event, where each element in the array is
        either a success (1) or a failure (0)
        :type output_data: np.ndarray
        :return: a `BinomTestResult` object.
        """

        # num_outputs = output_data.shape[0]
        proportion_inside_ci, k_inside, n_predictions = self.proportion_inside(
            prediction, output_data)
        binom_test = scipy.stats.binomtest(
            k_inside,  # number of successes from n bernoulli trials
            n_predictions,  # number of bernoulli trials
            self.confidence_interval,  # null hypothesis proportion
            alternative='two-sided')
        # dont reject null hypothesis if p-value is greater than alpha
        accept_stat = binom_test.pvalue >= self.alpha
        # confidence interval for proportion of successes
        stat_confidence_interval = np.array(binom_test.proportion_ci(confidence_level=1-self.alpha))
        return BinomTestResult(
            prop_inside=proportion_inside_ci,
            pvalue=binom_test.pvalue,
            prop_ci=stat_confidence_interval,
            accept_stat=accept_stat)

    def proportion_inside(self, prediction: typing.Union[np.ndarray, UnivariateGaussian], output_data: np.ndarray):
        """
        The function calculates the proportion of predictions that fall within a confidence interval.

        :param prediction: The `prediction` parameter can be either a numpy array or an instance of the
        `UnivariateGaussian` class. It represents the predicted values or the predicted probability
        distribution for a particular output variable
        :type prediction: typing.Union[np.ndarray, UnivariateGaussian]
        :param output_data: The `output_data` parameter is a numpy array that represents the observed
        output data. It should have shape `(num_outputs, num_predictions)`, where `num_outputs` is the
        number of output variables and `num_predictions` is the number of prediction samples. Each
        column of `output_data` represents
        :type output_data: np.ndarray
        :return: three values: proportion_inside_ci, k_inside, and n_predictions.
        """
        output_data = output_data.transpose()
        n_predictions = output_data.shape[1]

        two_tailed_percentiles = 100. * self._two_tailed_ci_levels

        if isinstance(prediction, UnivariateGaussian):
            percentiles = prediction.get_gaussian_quantiles(
                quantiles_lower_tail_probability=two_tailed_percentiles/100.)
        # samped predictions
        else:
            percentiles = np.percentile(prediction, two_tailed_percentiles, axis=0)
            # num_prediction_samples = prediction.shape[0] * num_outputs

        output_lower_than_lower_bound = np.sum(np.less(output_data[0, :], percentiles[0, :]))
        output_greater_than_upper_bound = np.sum(np.greater(output_data[0, :], percentiles[1, :]))

        # number of predictions inside confidence interval
        k_inside = n_predictions - \
            (output_lower_than_lower_bound + output_greater_than_upper_bound)
        proportion_inside_ci = k_inside / n_predictions

        return proportion_inside_ci, k_inside, n_predictions


@dataclass
class AneesTest(StatTest):
    """ Average Normalized Estimation Error Squared (ANEES) test

    The ANEES test is a statistical test for evaluating the performance of a filter. It is based on the
    Average Normalized Estimation Error Squared (ANEES) metric. 

    Uses Chi2 distribution to calculate confidence interval for ANEES.

    :param alpha: The `alpha` parameter is a float that represents the significance level of the test.
    :type alpha: float
    """
    # _alpha_lower: float
    # _alpha_upper: float

    _alpha_lower_upper: np.ndarray

    def __init__(self, alpha: float):
        # constructor of parent class
        super().__init__(alpha=alpha)

        alpha_lower = alpha / 2
        alpha_upper = 1 - alpha_lower

        self._alpha_lower_upper = np.array([alpha_lower, alpha_upper])

    def calc_anees_test(self, prediction: typing.Union[np.ndarray, UnivariateGaussian], output_data: np.ndarray) -> AneesTestResult:
        """
        The function calculates the Average Normalized Estimation Error Squared (ANEES) test for a given
        prediction and output data.

        :param prediction: The `prediction` parameter can be either a numpy array (`np.ndarray`) or an
        instance of the `UnivariateGaussian` class. It represents the predicted values or the estimated
        distribution of the output data
        :type prediction: typing.Union[np.ndarray, UnivariateGaussian]
        :param output_data: The `output_data` parameter is a numpy array that represents the actual
        output data. It has a shape of `(num_predictions, dim_output)`, where `num_predictions` is the
        number of predictions and `dim_output` is the dimension of the output
        :type output_data: np.ndarray
        :return: an instance of the `AneesTestResult` class. The `AneesTestResult` object contains the
        following attributes:
        """

        # number of predictions and dimension of output
        num_predictions, dim_output = output_data.shape

        # normalized estimation error squared (NEES)
        nees = squared_mahalanobis_distance(prediction, output_data)

        # average NEES
        anees = np.mean(nees)
        degrees_of_freedom = num_predictions * dim_output

        # confidence interval for NEES based on alpha, chi2 distribution, and degrees of freedom
        anees_confidence_interval = 1/num_predictions * \
            scipy.stats.chi2.ppf(self._alpha_lower_upper, degrees_of_freedom)

        accept_anees = anees_confidence_interval[0] <= anees <= anees_confidence_interval[1]
        sum_nees = anees * num_predictions
        pvalue_lower = scipy.stats.chi2.cdf(sum_nees, df=degrees_of_freedom)
        pvalue_upper = 1 - pvalue_lower
        pvalue = 2 * min([pvalue_lower, pvalue_upper])

        # check if NEES is chi2 distributed
        ks_stat = scipy.stats.ks_1samp(nees,
                                       scipy.stats.chi2.cdf, args=(dim_output,),
                                       alternative='two-sided')
        is_chi2 = ks_stat[1] >= self.alpha

        return AneesTestResult(
            anees=anees,
            anees_crit_bounds=anees_confidence_interval,
            pvalue=pvalue,
            accept_stat=accept_anees,
            nees_is_chi2=is_chi2)
