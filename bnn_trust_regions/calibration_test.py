
from dataclasses import dataclass
import typing

import numpy as np
import scipy.stats

from .distance_measures import squared_mahalanobis_distance
from .gaussian import UnivariateGaussian


@dataclass
class TestResult:
    accept_stat: bool
    pvalue: float


@dataclass
class BinomTestResult(TestResult):
    prop_inside: float
    # pvalue: float
    prop_ci: np.ndarray  # [0] is lower bound, [1] is upper bound
    # accept_stat: bool

    def __init__(self, prop_inside: float, pvalue: float, prop_ci: np.ndarray, accept_stat: bool):
        super().__init__(accept_stat=accept_stat, pvalue=pvalue)
        self.prop_inside = prop_inside
        # self.pvalue = pvalue
        self.prop_ci = prop_ci
        # self.accept_stat = accept_stat

    def prop_ci_low(self):
        return self.prop_ci[0]

    def prop_ci_high(self):
        return self.prop_ci[1]


class AneesTestResult(TestResult):
    anees: float
    # pvalue: float
    anees_crit_bounds: np.ndarray
    nees_is_chi2: bool
    # accept_stat: bool

    def __init__(self, anees: float, anees_crit_bounds: np.ndarray, pvalue: float, accept_stat: bool, nees_is_chi2: bool):
        super().__init__(accept_stat=accept_stat, pvalue=pvalue)
        self.anees = anees
        self.anees_crit_bounds = anees_crit_bounds
        self.nees_is_chi2 = nees_is_chi2

    def anees_crit_bound_low(self):
        return self.anees_crit_bounds[0]

    def anees_crit_bound_high(self):
        return self.anees_crit_bounds[1]


@dataclass
class StatTest:
    alpha: float


@dataclass
class BinomialTest(StatTest):
    confidence_interval: float  # e.g. 0.95 for testing of 95% confidence interval
    # e.g. [0.025 0.975] for testing of 95% confidence interval
    _two_tailed_ci_levels: np.ndarray

    # alpha: float  # e.g. 0.05 for significance level of 5%

    def __init__(self, confidence_interval: float, alpha: float):
        # constructor of parent class
        super().__init__(alpha=alpha)
        self.confidence_interval = confidence_interval
        # self.alpha = alpha

        two_tailed = (1 - self.confidence_interval) / 2
        self._two_tailed_ci_levels = np.array([two_tailed, 1-two_tailed])

    def calc_two_sided_binomial_test(self, prediction: typing.Union[np.ndarray, UnivariateGaussian], output_data: np.ndarray):

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
