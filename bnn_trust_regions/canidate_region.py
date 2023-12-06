""" class CandidateRegion """


from dataclasses import dataclass
import typing

import numpy as np

from prettytable import PrettyTable

from .gaussian import UnivariateGaussian
from .calibration_test import AneesTest, AneesTestResult, BinomialTest, BinomTestResult


@dataclass
class CandidateRegion:
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
    regions: typing.List[CandidateRegion]

    def __init__(self, regions: typing.List[CandidateRegion]):
        self.regions = regions

    def binomial_test(self, confidence_interval: float, alpha: float):

        prop_test = BinomialTest(confidence_interval=confidence_interval, alpha=alpha)
        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            prop_test_result = prop_test.calc_two_sided_binomial_test(
                prediction=region.predictions_in_region,
                output_data=region.outputs_in_region)
            region.binom_test_result = prop_test_result

    def anees_test(self, alpha: float, ):
        anees_test = AneesTest(alpha=alpha)

        for region in self.regions:
            assert isinstance(region, CandidateRegion)
            anees_test_result = anees_test.calc_anees_test(
                prediction=region.predictions_in_region,
                output_data=region.outputs_in_region)
            region.anees_test_result = anees_test_result

    def print_anees_test_results(self):

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

    def get_num_regions(self) -> int:
        return len(self.regions)

    def get_region(self, index: int) -> CandidateRegion:
        return self.regions[index]

    def get_region_by_x(self, x: float) -> CandidateRegion:
        for region in self.regions:
            if region.x_min <= x <= region.x_max:
                return region
        raise ValueError("No region found for x = {}".format(x))
