""" class CandidateRegion """


from dataclasses import dataclass
import typing

import numpy as np

from prettytable import PrettyTable

from .gaussian import UnivariateGaussian
from .calibration_test import AneesTest, AneesTestResult, BinomialTest, BinomTestResult


@dataclass
class CandidateRegion:
    """
    Represents a candidate region in the BNN trust regions algorithm.


    :param x_min: The minimum value of the candidate region.
    :type x_min: float
    :param x_max: The maximum value of the candidate region.
    :type x_max: float

    :param predictions_in_region: The predictions within the candidate region.
    :type predictions_in_region: Union[np.ndarray, UnivariateGaussian]
    :param outputs_in_region: The outputs within the candidate region.
    :type outputs_in_region: np.ndarray

    :param binom_test_result: The result of the binomial test.
    :type binom_test_result: BinomTestResult
    :param anees_test_result: The result of the ANEES test.
    :type anees_test_result: AneesTestResult
    """

    x_min: float
    x_max: float

    predictions_in_region: typing.Union[np.ndarray, UnivariateGaussian]
    outputs_in_region: np.ndarray

    binom_test_result: BinomTestResult
    anees_test_result: AneesTestResult

    def __init__(self, predictions_in_region: typing.Union[np.ndarray, UnivariateGaussian],
                 outputs_in_region: np.ndarray,
                 x_min: float,
                 x_max: float,):

        # if x_min/x_max is numpy array, convert to float
        if isinstance(x_min, np.ndarray):
            x_min = x_min.item()
        if isinstance(x_max, np.ndarray):
            x_max = x_max.item()

        self.x_min = x_min
        self.x_max = x_max

        self.predictions_in_region = predictions_in_region
        self.outputs_in_region = outputs_in_region


@dataclass
class CandidateRegions:
    """
        The CandidateRegions class represents a collection of candidate regions.
    """
    regions: typing.List[CandidateRegion]

    def __init__(self, regions: typing.List[CandidateRegion]):
        """
        The CandidateRegions class represents a collection of candidate regions.

        :param regions: A list of CandidateRegion objects representing the candidate regions.
        :type regions: typing.List[CandidateRegion]
        """
        self.regions = regions

    def binomial_test(self, confidence_interval: float, alpha: float):
        """
        The function performs a two-sided binomial test on the predictions and outputs of candidate
        regions.

        :param confidence_interval: The confidence interval is a measure of the uncertainty associated
        with an estimate. It represents the range within which the true value is likely to
        fall with a certain level of confidence.
        In this context, it is used as a parameter for the
        binomial test to test the confidence interval for the proportion of test data.
        :type confidence_interval: float
        :param alpha: The alpha parameter is the significance level, which is the probability of
        rejecting the null hypothesis when it is true. It is typically set to a value between 0 and 1,
        such as 0.05 or 0.01.
        :type alpha: float
        """

        prop_test = BinomialTest(confidence_interval=confidence_interval, alpha=alpha)
        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            prop_test_result = prop_test.calc_two_sided_binomial_test(
                prediction=region.predictions_in_region,
                output_data=region.outputs_in_region)
            region.binom_test_result = prop_test_result

    def anees_test(self, alpha: float, ):
        """
        The function `anees_test` calculates the ANEES test result for each region in a list of
        candidate regions.

        :param alpha: The alpha parameter is a float value that represents the significance level for
        the Anees test. It is used to determine the critical value for the test statistic.
        :type alpha: float
        """
        anees_test = AneesTest(alpha=alpha)

        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            anees_test_result = anees_test.calc_anees_test(
                prediction=region.predictions_in_region,
                output_data=region.outputs_in_region)
            region.anees_test_result = anees_test_result

    def print_anees_test_results(self):
        """
        The function `print_anees_test_results` prints the ANEES test results in a table format.
        """

        # print as table in console
        anees_test_table = PrettyTable()
        anees_test_table.field_names = ["x_min", "x_max", "anees", "p-value",
                                        "anees crit bound low", "anees crit bound high",  "calibrated predictions", "nees is chi2"]
        anees_test_table.float_format = ".2"
        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            anees_test_table.add_row([region.x_min, region.x_max,
                                      region.anees_test_result.anees,
                                      region.anees_test_result.pvalue,
                                      region.anees_test_result.anees_crit_bound_low(),
                                      region.anees_test_result.anees_crit_bound_high(),
                                      region.anees_test_result.accept_stat,
                                      region.anees_test_result.nees_is_chi2])
        print('ANEES test results:')
        print(anees_test_table)

    def print_binomial_test_results(self):
        """
        The function `print_binomial_test_results` prints the results of a binomial test in a table
        format.
        """

        # print as table in console
        binom_test_table = PrettyTable()
        binom_test_table.field_names = ["x_min", "x_max", "proportion inside",
                                        "p-value", "prop CI low", "prop CI high",  "calibrated predictions"]
        binom_test_table.float_format = ".2"
        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            binom_test_table.add_row([region.x_min, region.x_max,
                                      region.binom_test_result.prop_inside,
                                      region.binom_test_result.pvalue,
                                      region.binom_test_result.prop_ci_low(),
                                      region.binom_test_result.prop_ci_high(),
                                      region.binom_test_result.accept_stat])
        print('Binomial test results:')
        print(binom_test_table)

    def get_binom_results(self):
        """
        The function ``get_binom_results`` returns the test results of the binomial test
        as numpy arrays.

        :return: A tuple containing the ANEES test results as numpy arrays.
            prop_inside (numpy.ndarray):
                Array of proportions inside each region.
            binom_deviation_from_stat (numpy.ndarray):
                Array of deviations from the binomial test statistic for each region.
            tested_proportion (float):
                Proportion used in the binomial test.
            pvalues (numpy.ndarray):
                Array of p-values for each region.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        prop_inside = np.zeros((len(self.regions), ))
        binom_deviation_from_stat = np.zeros((2, len(self.regions)))
        pvalues = np.zeros((len(self.regions), ))
        for idx, region in enumerate(self.regions):
            # binomial test result
            prop_inside[idx] = region.binom_test_result.prop_inside
            lower_deviation = prop_inside[idx] - region.binom_test_result.prop_ci_low()
            upper_deviation = region.binom_test_result.prop_ci_high() - prop_inside[idx]
            binom_deviation_from_stat[:, idx] = np.array([lower_deviation, upper_deviation])
            tested_proportion = self.regions[0].binom_test_result.tested_proportion
            pvalues[idx] = region.binom_test_result.pvalue
        return prop_inside, binom_deviation_from_stat, tested_proportion, pvalues

    def get_anees_results(self):
        """
        The function `get_anees_results` returns the ANEES test results as numpy arrays.

        :return: A tuple containing the ANEES test results as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        anees_stat = np.zeros((len(self.regions), ))
        anees_crit_deviations_from_1 = np.zeros((2, len(self.regions)))
        nees_is_chi2 = np.zeros((len(self.regions), ), dtype=bool)
        pvalues = np.zeros((len(self.regions), ))
        for idx, region in enumerate(self.regions):
            anees_stat[idx] = region.anees_test_result.anees
            anees_crit_deviations_from_1[0, idx] = 1 - \
                region.anees_test_result.anees_crit_bound_low()
            anees_crit_deviations_from_1[1,
                                         idx] = region.anees_test_result.anees_crit_bound_high() - 1
            nees_is_chi2[idx] = region.anees_test_result.nees_is_chi2
            pvalues[idx] = region.anees_test_result.pvalue
        return anees_stat, anees_crit_deviations_from_1, nees_is_chi2, pvalues

    def get_anees_results_as_array(self):
        """
        The function `get_anees_results_as_array` returns the ANEES test results as a numpy array.

        :return: A numpy array containing the ANEES test results.
        :rtype: np.ndarray
        """
        anees_values = np.zeros((len(self.regions), ))
        anees_crit_bounds = np.zeros((len(self.regions), 2))
        input_range = np.zeros((len(self.regions), 2))
        nees_is_chi2 = np.zeros((len(self.regions), ), dtype=bool)
        pvalues = np.zeros((len(self.regions), ))

        for idx, region in enumerate(self.regions):
            assert isinstance(region, CandidateRegion)
            anees_values[idx] = region.anees_test_result.anees
            anees_crit_bounds[idx, 0] = region.anees_test_result.anees_crit_bound_low()
            anees_crit_bounds[idx, 1] = region.anees_test_result.anees_crit_bound_high()
            nees_is_chi2[idx] = region.anees_test_result.nees_is_chi2
            pvalues[idx] = region.anees_test_result.pvalue
            input_range[idx, 0] = region.x_min
            input_range[idx, 1] = region.x_max

        return anees_values, anees_crit_bounds, nees_is_chi2, pvalues, input_range

    def get_num_regions(self) -> int:
        """
        The function returns the number of regions in an object.

        :return: The number of regions in the object.
        :rtype: int
        """
        return len(self.regions)

    def get_region(self, index: int) -> CandidateRegion:
        """
        The function `get_region` returns a `CandidateRegion` object based on the given index.

        :param index: The index parameter is an integer that represents the position of the candidate
        region in the list of regions.
        :type index: int
        :return: A CandidateRegion object.
        :rtype: CandidateRegion
        """
        return self.regions[index]

    def get_region_by_x(self, x: float) -> CandidateRegion:
        """
        The function `get_region_by_x` returns the candidate region that contains a given x-coordinate.

        :param x: A float value representing the x-coordinate of a point.
        :type x: float
        :return: A CandidateRegion object.
        :rtype: CandidateRegion
        """
        for region in self.regions:
            if region.x_min <= x <= region.x_max:
                return region
        raise ValueError(f"No region found for x = {x}")
