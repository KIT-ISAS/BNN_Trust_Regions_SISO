""" Class for evaluating models by comparing a reference model to an approximation model."""


import dataclasses
import enum
import typing


import numpy as np

from .canidate_region import CandidateRegion, CandidateRegions
from .candidate_region_identification import SisoCandidateRegionIdentification
from .gaussian import UnivariateGaussian
from .io_data import IOData
from .wasserstein_dist import WassersteinDistance
from .stat_test_settings import StatTestSettings


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
    candidate_regions: CandidateRegions  # candidate regions

    # critical_distance: float  # critical distance
    wasserstein: WassersteinDistance  # instance to calculate the Wasserstein distance

    stat_test_settings: None  # create dict or class for settings  # settings for statistical tests

    def __init__(self,
                 predictions_a: typing.Union[np.ndarray, UnivariateGaussian],
                 predictions_b: typing.Union[np.ndarray, UnivariateGaussian],
                 test_data: IOData,
                 wasserstein_distance: WassersteinDistance = None,):

        self.predictions_a = predictions_a
        self.predictions_b = predictions_b
        self.test_data = test_data  # TODO sort according to input values
        self.num_distributions = test_data.output.shape[0]

        self.set_distance_settings(wasserstein_distance)

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

    def calc_statistical_tests(self, stat_test_settings: StatTestSettings, use_a_or_b: UseAorB = UseAorB.B):

        alpha = stat_test_settings.alpha
        confidence_interval = stat_test_settings.confidence_interval

        if use_a_or_b == UseAorB.A:
            self.candidate_regions = self.region_ident.split_data_in_regions(self.predictions_a)
        elif use_a_or_b == UseAorB.B:
            self.candidate_regions = self.region_ident.split_data_in_regions(self.predictions_b)
        else:
            raise ValueError("use_a_or_b must be either UseAorB.A or UseAorB.B")

        self.candidate_regions.binomial_test(confidence_interval=confidence_interval, alpha=alpha)
        self.candidate_regions.anees_test(alpha=alpha)

    def print_statistical_tests(self):
        self.candidate_regions.print_binomial_test_results()
        self.candidate_regions.print_anees_test_results()

    def evaluate(self):
        # Add your evaluation logic here
        pass


# if __name__ == "__main__":
#     # test some functionality

#     np.random.seed(42)
#     pred_a1 = np.random.randn(1000, 100)  # 1000 samples for each input value, 100 output
#     pred_b1 = np.random.randn(1000, 100)  # 1000 samples for each input value, 100 output

#     data = np.random.randn(1000, 1)  # 1000 samples for each input value, 1 output

#     ws_dist_settings1 = WassersteinDistance(p_norm=1, parallel_computing=True, verbose=False)
#     model_evaluator1 = ModelEvaluator(
#         predictions_a=pred_a1, predictions_b=pred_b1, wasserstein_distance=ws_dist_settings1, test_data=data)

#     model_evaluator1.calc_wasserstein_distance()

#     pred_a2 = UnivariateGaussian(mean=np.mean(pred_a1, axis=0), var=np.var(pred_a1, axis=0))
#     pred_b2 = UnivariateGaussian(mean=np.mean(pred_b1, axis=0), var=np.var(pred_a1, axis=0))

#     ws_dist_settings2 = WassersteinDistance(p_norm=1, parallel_computing=False, verbose=False)
#     model_evaluator2 = ModelEvaluator(
#         predictions_a=pred_a2, predictions_b=pred_b2, test_data=data)
#     model_evaluator2.set_distance_settings(ws_dist_settings2)
#     model_evaluator2.calc_wasserstein_distance()

#     # difference reduces if more samples are used # looks plausbiel
#     diff = model_evaluator2.distance - model_evaluator1.distance
#     print(diff)
