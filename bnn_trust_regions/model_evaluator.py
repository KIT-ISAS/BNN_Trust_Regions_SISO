""" Class for evaluating models by comparing a reference model to an approximation model."""


import dataclasses
import typing


import numpy as np

from .candidate_region_identification import CandidateRegionIdentification


from .gaussian import UnivariateGaussian
from .io_data import IOData
from .wasserstein_dist import WassersteinDistance


@dataclasses.dataclass  # needed?
class ModelEvaluator:
    """Class for evaluating models by comparing a reference model to an approximation model."""
    predictions_a: typing.Union[np.ndarray, UnivariateGaussian]  # reference model
    predictions_b: typing.Union[np.ndarray, UnivariateGaussian]  # approximation model

    test_data: IOData  # test data, IOData Instance
    num_distributions: int  # number of distributions

    distance: np.ndarray  # distance information

    region_ident: CandidateRegionIdentification  # instance for candidate region identification

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
        self.test_data = test_data  # TODO sort input values
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

    def calc_canidate_regions(self, region_ident: CandidateRegionIdentification):
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

        self.split_data_in_regions(self.predictions_a)

        # self.region_ident.extendend_switching_points
        # self.region_ident.switching_points

    def split_data_in_regions(self, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                              ) -> typing.Tuple[typing.List[int], typing.List[int]]:
        """
        The function `split_in_local_clusters` splits prediction and output data into valid and invalid
        intervals based on an invalid range.

        :param prediction: An array of predictions
        :param output_data: The `output_data` parameter is an array of output data. It is of type
        `numpy.ndarray`
        :type output_data: np.ndarray
        :param invalid_range: An array of boolean values indicating invalid intervals. Each element in the
        array corresponds to a prediction, and a value of True indicates that the prediction is invalid
        :type invalid_range: np.ndarray
        :param min_points_per_cluster: The parameter `min_points_per_cluster` is an integer that specifies
        the minimum number of points required for a cluster to be considered valid. If a cluster has fewer
        points than this threshold, it will be considered invalid
        :type min_points_per_cluster: int
        :return: a tuple containing three elements: `splited_prediction`, `splited_output`, and
        `extended_switching_range`.
        """

        num_predictions = self.num_distributions
        output_data = self.test_data.output

        valid_invalid_switching = self.region_ident.switching_points
        extended_switching_range = self.region_ident.extendend_switching_points

        # if all prediction are valid or invalid -> dont split
        if 0 < valid_invalid_switching.size < output_data.size:

            splited_output = np.split(output_data, valid_invalid_switching, axis=0)
            if isinstance(predictions, UnivariateGaussian):

                splitted_mean = np.split(predictions.mean, valid_invalid_switching, axis=0)
                splitted_var = np.split(predictions.var, valid_invalid_switching, axis=0)
                splited_prediction = []
                for _, (mean, var) in enumerate(zip(splitted_mean, splitted_var)):
                    splited_prediction.append(UnivariateGaussian(mean=mean, var=var))

            else:
                splited_prediction = np.split(predictions, valid_invalid_switching, axis=1)
        else:
            splited_prediction = [predictions]
            splited_output = [output_data]
            self.region_ident.extendend_switching_points = [0, num_predictions-1]
        # TODO use class instead of List?

        return splited_prediction, splited_output

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
