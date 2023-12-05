import numpy as np

import bnn_trust_regions

from bnn_trust_regions.model_evaluator import ModelEvaluator
from bnn_trust_regions.wasserstein_dist import WassersteinDistance
from bnn_trust_regions.gaussian import UnivariateGaussian
from bnn_trust_regions.io_data import IOData
from bnn_trust_regions.candidate_region_identification import IdentGifSettings, CandidateRegionIdentification

from bnn_trust_regions.utils import save_load


def test_functionality():

    np.random.seed(42)

    # folder with some example data and predictions
    data_folder = "example_data"
    # load test data
    test_data_file_name = "nn_test"
    test_input, test_output = save_load.load_io_data(data_folder, test_data_file_name)
    test_data = IOData(input=test_input, output=test_output)

    # load training data
    train_data_file_name = "nn_train"
    train_input, train_output = save_load.load_io_data(data_folder, train_data_file_name)
    train_data = IOData(input=train_input, output=train_output)

    # load predictions
    mcmc_file_name = "mcmc_test"
    pred_mcmc = save_load.load_sampled_predictions(data_folder, mcmc_file_name)
    svi_file_name = "meanfield_svi_test"
    pred_svi = save_load.load_sampled_predictions(data_folder, svi_file_name)
    pbp_file_name = "pbp_test"
    pred_pbp_mean, pred_pbp_var = save_load.load_sampled_predictions(data_folder, pbp_file_name)
    pred_pbp = UnivariateGaussian(mean=pred_pbp_mean, var=pred_pbp_var)

    # evaluate predictions from mcmc and svi
    pred_a = pred_mcmc  # mcmc as reference model
    pred_b1 = pred_svi  # svi as approximation model

    # set wasserstein distance settings
    ws_dist_settings1 = WassersteinDistance(p_norm=1, parallel_computing=True, verbose=False)

    # evaluate predictions from mcmc and svi
    model_evaluator1 = ModelEvaluator(
        predictions_a=pred_a, predictions_b=pred_b1, wasserstein_distance=ws_dist_settings1, test_data=test_data)

    # wasserstein distance between reference and approximation model
    model_evaluator1.calc_wasserstein_distance()

    gif_settings = IdentGifSettings(
        path="eval1_plots", file_name="crit_dist", dpi=200, fps=2, loop=0)
    # calculate candidate regions and plot gif of critical distance
    region_ident = CandidateRegionIdentification(
        min_points_per_region=200, smoothing_window_size=50, verbose=True, gif_settings=gif_settings)
    model_evaluator1.calc_canidate_regions(region_ident=region_ident)

    # evaluate predictions from mcmc and pbp
    # mcmc as reference model
    pred_b2 = pred_pbp  # pbp as approximation model

    ws_dist_settings2 = WassersteinDistance(p_norm=1, parallel_computing=False, verbose=False)
    model_evaluator2 = ModelEvaluator(
        predictions_a=pred_a, predictions_b=pred_b2, test_data=test_data)
    model_evaluator2.set_distance_settings(ws_dist_settings2)
    model_evaluator2.calc_wasserstein_distance()


if __name__ == "__main__":
    # test some functionality
    test_functionality()
