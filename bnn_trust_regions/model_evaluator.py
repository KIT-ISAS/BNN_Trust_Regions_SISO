""" Class for evaluating models by comparing a reference model to an approximation model."""


import copy
import dataclasses
import enum
import typing


import numpy as np

from bnn_trust_regions.utils.ci_prediction import calc_mean_and_quantiles

from .plot_candidate_regions import PlotSettings, PlotSisoCandidateRegions

from .canidate_region import CandidateRegions
from .candidate_region_identification import SisoCandidateRegionIdentification
from .gaussian import UnivariateGaussian
from .io_data import IOData
from .wasserstein_dist import WassersteinDistance
from .stat_test_settings import StatTestSettings

from .utils.sort_predictions import sort_predictions


class UseAorB(enum.Enum):
    """Enum for selecting which model to use for calculating the statistics per regions"""
    A = 1
    B = 2


@dataclasses.dataclass  # needed?
class ModelEvaluator:
    """Class for evaluating models by comparing a reference model to an approximation model."""
    predictions_a: typing.Union[np.ndarray, UnivariateGaussian]  # reference model
    predictions_b: typing.Union[np.ndarray, UnivariateGaussian]  # approximation model

    test_data: IOData  # test data, IOData Instance
    num_distributions: int  # number of distributions

    distance: np.ndarray  # distance information

    region_ident: SisoCandidateRegionIdentification  # instance for candidate region identification
    candidate_regions_model_a: CandidateRegions  # candidate regions
    candidate_regions_model_b: CandidateRegions  # candidate regions

    # critical_distance: float  # critical distance
    wasserstein: WassersteinDistance  # instance to calculate the Wasserstein distance

    stat_test_settings: None  # create dict or class for settings  # settings for statistical tests

    def __init__(self,
                 predictions_a: typing.Union[np.ndarray, UnivariateGaussian],
                 predictions_b: typing.Union[np.ndarray, UnivariateGaussian],
                 test_data: IOData,
                 wasserstein_distance: WassersteinDistance = None,):

        self.predictions_a = copy.deepcopy(predictions_a)
        self.predictions_b = copy.deepcopy(predictions_b)
        self.test_data = copy.deepcopy(test_data)
        self.num_distributions = test_data.output.shape[0]

        # sort according to input values
        self.sort_test_data_and_predictions()

        self.set_distance_settings(wasserstein_distance)

    def sort_test_data_and_predictions(self):
        """
        The function sorts the test data and predictions according to the input values.
        """
        # sort input values and get indices
        idx = np.argsort(self.test_data.input, axis=None)
        # is idx equal to np.arange(len(idx))?
        if np.array_equal(idx, np.arange(len(idx))):
            # no need to sort
            return

        # sort predictions according to indices
        self.test_data.input = self.test_data.input[idx]
        self.test_data.output = self.test_data.output[idx]

        # sort deep copy of predictions
        self.predictions_a = sort_predictions(self.predictions_a, idx)
        self.predictions_b = sort_predictions(self.predictions_b, idx)

    def set_distance_settings(self,
                              wasserstein_distance: WassersteinDistance,
                              p_norm: int = 1,
                              parallel_computing: bool = True,
                              verbose: bool = False):
        """
        The function sets the distance settings for the Wasserstein distance metric, allowing for
        customization of the p-norm, parallel computing, and verbosity.

        :param wasserstein_distance: The `wasserstein_distance` parameter is an instance of the
        `WassersteinDistance` class. It represents the distance metric used for calculating the Wasserstein
        distance. If no custom distance metric is provided, the method will use the default settings and
        create a new instance of the `Wasserstein
        :type wasserstein_distance: WassersteinDistance
        :param p_norm: The p_norm parameter is an integer that specifies the norm to be used in the
        Wasserstein distance calculation. It determines the type of distance metric used. The default value
        is 1.
        :type p_norm: int (optional)
        :param parallel_computing: The `parallel_computing` parameter is a boolean flag that determines
        whether or not to use parallel computing for calculating the Wasserstein distance. If set to `True`,
        parallel computing will be used, which can potentially speed up the computation. If set to `False`,
        the computation will be done sequentially, defaults to True
        :type parallel_computing: bool (optional)
        :param verbose: The `verbose` parameter is a boolean flag that determines whether or not to print
        additional information during the computation. If `verbose` is set to `True`, then additional
        information will be printed. If `verbose` is set to `False`, then no additional information will be
        printed, defaults to False
        :type verbose: bool (optional)
        :return: nothing.
        """
        # default Wasserstein distance settings
        if wasserstein_distance is None:
            self.wasserstein = WassersteinDistance(
                p_norm, parallel_computing, verbose)
            return
        # custom Wasserstein distance settings
        self.wasserstein = wasserstein_distance

    def calc_wasserstein_distance(self):
        """
        The function calculates the Wasserstein distance between two sets of predictions.
        """
        if self.wasserstein is None:
            raise ValueError("Wasserstein distance settings are not set.")
        self.distance = self.wasserstein.calc_wasserstein_distance(
            self.predictions_a, self.predictions_b)

    def calc_canidate_regions(self, region_ident: SisoCandidateRegionIdentification):
        """
        The function calculates the candidate regions based on th Wasserstein distance.
        """
        self.region_ident = region_ident

        if region_ident.raw_distances is None:
            self.region_ident.raw_distances = self.distance
        if region_ident.test_data is None:
            self.region_ident.test_data = self.test_data

        self.region_ident.smooth_distances()
        self.region_ident.calc_critical_distance()
        self.region_ident.subsplit_candidate_regions()

    def calc_statistical_tests(self, stat_test_settings: StatTestSettings,

                               ):
        """
        The function calculates statistical tests on candidate regions based on the given settings and
        predictions.

        :param stat_test_settings: The `stat_test_settings` parameter is an instance of the
        `StatTestSettings` class. It contains the settings for the statistical tests, including the
        alpha value (significance level) and the confidence interval which should be tested by the binomial test.
        :type stat_test_settings: StatTestSettings
        :param use_a_or_b: The parameter "use_a_or_b" is an optional parameter that determines whether
        to use the predictions from group A or group B for the statistical tests. It can take two
        possible values: UseAorB.A or UseAorB.B
        :type use_a_or_b: UseAorB
        """

        alpha = stat_test_settings.alpha
        confidence_interval = stat_test_settings.confidence_interval

        self.candidate_regions_model_a = self.region_ident.split_data_in_regions(self.predictions_a)
        self.candidate_regions_model_a.binomial_test(
            confidence_interval=confidence_interval, alpha=alpha)
        self.candidate_regions_model_a.anees_test(alpha=alpha)

        self.candidate_regions_model_b = self.region_ident.split_data_in_regions(self.predictions_b)
        self.candidate_regions_model_b.binomial_test(
            confidence_interval=confidence_interval, alpha=alpha)
        self.candidate_regions_model_b.anees_test(alpha=alpha)

    def print_statistical_tests(self, model_names: typing.Tuple[str, str] = ('reference model', 'approximation model')):
        """
        The function "print_statistical_tests" prints the results of binomial and ANEES tests for each
        candidate regions to console.
        """
        self._print_one_model_stat_test(
            candidate_regions=self.candidate_regions_model_a, model_name=model_names[0])
        self._print_one_model_stat_test(
            candidate_regions=self.candidate_regions_model_b, model_name=model_names[1])

    def plot_statistical_tests(self, plot_settings: PlotSettings, ground_truth:
                               typing.Union[np.ndarray, UnivariateGaussian] = None,
                               model_names: typing.Tuple[str, str] = ('MCMC', 'SVI')):
        """
        The function "plot_statistical_tests" plots the results of binomial and ANEES tests for
        candidate regions.
        """

        # plot settings for model a and b
        plot_settings_a = copy.deepcopy(plot_settings)
        plot_settings_a.model_name = model_names[0] + '_a'
        plot_settings_b = copy.deepcopy(plot_settings)
        plot_settings_b.model_name = model_names[1] + '_b'

        self._plot_one_model(candidate_regions=self.candidate_regions_model_a,
                             predictions=self.predictions_a,
                             plot_settings=plot_settings_a,
                             ground_truth=ground_truth)

        self._plot_one_model(candidate_regions=self.candidate_regions_model_b,
                             predictions=self.predictions_b,
                             plot_settings=plot_settings_b,
                             ground_truth=ground_truth)

    def _print_one_model_stat_test(self, candidate_regions: CandidateRegions, model_name: str):
        print(f"Model {model_name}:")
        candidate_regions.print_binomial_test_results()
        print(f"Model {model_name}:")
        candidate_regions.print_anees_test_results()

    def _plot_one_model(self, candidate_regions: CandidateRegions, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                        plot_settings: PlotSettings, ground_truth: typing.Union[np.ndarray, UnivariateGaussian] = None):
        """"""

        plot_instance = PlotSisoCandidateRegions(
            candidate_regions=candidate_regions,
            plot_settings=plot_settings,)

        if ground_truth is None:
            ground_truth_mean = None
        else:
            ground_truth_mean, _ = calc_mean_and_quantiles(
                ground_truth, plot_settings.confidence_interval)

        plot_instance.plot_predictions_with_region_results(
            predictions=predictions,
            data=self.test_data,
            ground_truth=ground_truth_mean,
        )
        plot_instance.plot_stats_per_region()
        plot_instance.plot_stats_and_predictions(data=self.test_data,
                                                 predictions=predictions,)

        if not isinstance(ground_truth, UnivariateGaussian):
            return

        ws_dist_gt = self.wasserstein.calc_wasserstein_distance(
            predictions, ground_truth).reshape(1, -1)
        plot_instance.plot_stats_and_ground_truth_dist(data=self.test_data,
                                                       dist_to_ground_truth=ws_dist_gt)
