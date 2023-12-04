""" Class for evaluating models by comparing a reference model to an approximation model."""

import dataclasses
from pyexpat import model
import typing


import numpy as np


from gaussian import UnivariateGaussian
from wasserstein_dist import WassersteinDistance


@dataclasses.dataclass
class ModelEvaluator:
    """Class for evaluating models by comparing a reference model to an approximation model."""
    predictions_a: typing.Union[np.ndarray, UnivariateGaussian]  # reference model
    predictions_b: typing.Union[np.ndarray, UnivariateGaussian]  # approximation model
    num_distributions: int

    distance: np.ndarray  # distance information
    wasserstein: WassersteinDistance  # instance to calculate the Wasserstein distance

    def __init__(self,
                 predictions_a: typing.Union[np.ndarray, UnivariateGaussian],
                 predictions_b: typing.Union[np.ndarray, UnivariateGaussian],
                 wasserstein_distance: WassersteinDistance = None,):
        self.predictions_a = predictions_a
        self.predictions_b = predictions_b

        self.set_distance_settings(wasserstein_distance)

    def set_distance_settings(self, wasserstein_distance: WassersteinDistance, p_norm: int = 1, parallel_computing: bool = True, verbose: bool = False):
        # default Wasserstein distance settings
        if wasserstein_distance is None:
            self.wasserstein = WassersteinDistance(
                p_norm, parallel_computing, verbose)
            return
        # custom Wasserstein distance settings
        self.wasserstein = wasserstein_distance

    def calc_wasserstein_distance(self):
        if self.wasserstein is None:
            raise ValueError("Wasserstein distance settings are not set.")
        self.distance = self.wasserstein.calc_wasserstein_distance(
            self.predictions_a, self.predictions_b)

    def evaluate(self):
        # Add your evaluation logic here
        pass


if __name__ == "__main__":
    # test some functionality

    np.random.seed(42)
    pred_a1 = np.random.randn(1000, 100)  # 1000 samples for each input value, 100 output
    pred_b1 = np.random.randn(1000, 100)  # 1000 samples for each input value, 100 output

    ws_dist_settings1 = WassersteinDistance(p_norm=1, parallel_computing=True, verbose=False)
    model_evaluator1 = ModelEvaluator(
        predictions_a=pred_a1, predictions_b=pred_b1, wasserstein_distance=ws_dist_settings1)

    model_evaluator1.calc_wasserstein_distance()

    pred_a2 = UnivariateGaussian(mean=np.mean(pred_a1, axis=0), var=np.var(pred_a1, axis=0))
    pred_b2 = UnivariateGaussian(mean=np.mean(pred_b1, axis=0), var=np.var(pred_a1, axis=0))

    ws_dist_settings2 = WassersteinDistance(p_norm=1, parallel_computing=False, verbose=False)
    model_evaluator2 = ModelEvaluator(predictions_a=pred_a2, predictions_b=pred_b2)
    model_evaluator2.set_distance_settings(ws_dist_settings2)
    model_evaluator2.calc_wasserstein_distance()

    # difference reduces if more samples are used # looks plausbiel
    diff = model_evaluator2.distance - model_evaluator1.distance
    print(diff)
