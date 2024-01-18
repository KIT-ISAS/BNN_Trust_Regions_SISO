""" Test some functionality of the bnn_trust_regions package."""


# from email.mime import image
# from os import error
# from matplotlib.pyplot import errorbar
import numpy as np

from bnn_trust_regions.candidate_region_identification import IdentGifSettings, SisoCandidateRegionIdentification
from bnn_trust_regions.gaussian import UnivariateGaussian
from bnn_trust_regions.io_data import IOData
from bnn_trust_regions.model_evaluator import ModelEvaluator
from bnn_trust_regions.plot_candidate_regions import ErrorbarPlotSettings, PlotSettings, DistributionPlotSettings
from bnn_trust_regions.stat_test_settings import StatTestSettings
from bnn_trust_regions.utils import save_load, matplotlib_settings
from bnn_trust_regions.wasserstein_dist import WassersteinDistance

np.random.seed(42)
matplotlib_settings.init_settings(use_tex=True)


def test_functionality():
    """ Test some functionality of the bnn_trust_regions package."""

    ########################################################################################################
    # load test data
    ########################################################################################################
    # folder with some example data and predictions
    data_folder = "example_data"
    # load test data
    test_data_file_name = "nn_test"
    test_input, test_output = save_load.load_io_data(data_folder, test_data_file_name)
    test_data = IOData(input=test_input, output=test_output)

    # load training data
    # train_data_file_name = "nn_train"
    # train_input, train_output = save_load.load_io_data(data_folder, train_data_file_name)
    # train_data = IOData(input=train_input, output=train_output)

    # load predictions
    mcmc_file_name = "mcmc_test"
    pred_mcmc = save_load.load_sampled_predictions(data_folder, mcmc_file_name)
    svi_file_name = "meanfield_svi_test"
    pred_svi = save_load.load_sampled_predictions(data_folder, svi_file_name)
    pbp_file_name = "pbp_test"
    pred_pbp_mean, pred_pbp_var = save_load.load_sampled_predictions(data_folder, pbp_file_name)
    pred_pbp = UnivariateGaussian(mean=pred_pbp_mean, var=pred_pbp_var)
    ########################################################################################################

    ########################################################################################################
    #  change predictions and test data and hyperparameters
    # for region identification and statistical testing here
    ########################################################################################################
    # set wasserstein distance settings
    p_norm = 1
    parallel_computing = True
    verbose = False
    ws_dist_settings1 = WassersteinDistance(
        p_norm=p_norm, parallel_computing=parallel_computing, verbose=verbose)

    # candidate region identification settings
    min_points_per_region = 200
    smoothing_window_size = 50
    plot_gif = True

    # display gif of regions identification critical distance
    plot_folder1 = "eval1_plots"
    plot_folder2 = "eval2_plots"
    file_name = "crit_dist"
    dpi = 200
    fps = 2
    loop = 0  # 0 for infinite loop
    gif_settings1 = IdentGifSettings(
        path=plot_folder1, file_name=file_name, dpi=dpi, fps=fps, loop=loop)

    region_ident1 = SisoCandidateRegionIdentification(
        min_points_per_region=min_points_per_region, smoothing_window_size=smoothing_window_size,
        verbose=plot_gif, gif_settings=gif_settings1)

    # statistical test settings
    alpha = 0.01  # significance level of 1%
    confidence_interval = 0.95  # test the 95% confidence interval
    stat_test_settings = StatTestSettings(alpha=alpha, confidence_interval=confidence_interval)

    # test model A or B
    ########################################################################################################

    ########################################################################################################
    # ground truth used for plotting
    ########################################################################################################
    # mean ground truth
    ground_truth_mean = np.power(test_data.input, 3).reshape(1, -1)
    ground_truth_distribution = UnivariateGaussian(mean=ground_truth_mean.squeeze(), var=9.)
    ########################################################################################################

    ########################################################################################################
    # set plot settings
    # more Settings can be found in the classes:
    #   ``PlotSettings`` ``ErrorbarPlotSettings`` ``DistributionPlotSettings``
    ########################################################################################################
    error_bar_plot_settings = ErrorbarPlotSettings(
        # label
        anees_label=r'ANEES is $\chi^2$',
        anees_label_notchi2=r'ANEES is not $\chi^2$',
        annes_errorbar_label='ANEES Bounds',
        split_label='Region Split',
        out_of_scope_label='ANEES o.s.',
        anees_y_label='ANEES',
        # colors
        anees_bar_color='tab:orange',
        anees_marker_color='tab:blue',
        binom_bar_color='tab:purple',
        binom_marker_color='tab:red',
    )
    plot_settings = PlotSettings(
        image_format='svg',  # image format of the plots
        # plot_folder='eval1_plots',  # folder where the plots are saved
        confidence_interval=0.95,
        # settings to plot the predictions and the mean of the ground truth
        ground_truth_plot_settings=DistributionPlotSettings(
            mean_color='tab:orange',
            mean_linestyle='-',
            mean_label=r'$y=x^3$',),  # label for mean value of ground truth
        wasserstein_plot_settings=DistributionPlotSettings(
            mean_color='k',
            mean_linestyle='-',
            mean_label=r'$W_1^\text{GT}$',),  # label for wasserstein distance between ground truth distribution and predictive distribution
        error_bar_plot_settings=error_bar_plot_settings,
    )

    ########################################################################################################
    # only testing
    # shuffling test data and their predictions
    ########################################################################################################
    # num_test_points = len(test_data.input)
    # idx = np.arange(num_test_points)
    # np.random.shuffle(idx)

    # test_data.input = test_data.input[idx]
    # test_data.output = test_data.output[idx]

    # pred_a = pred_a[:, idx]
    # pred_b1 = pred_b1[:, idx]
    # pred_b2 = UnivariateGaussian(mean=pred_b2.mean[idx], var=pred_b2.var[idx])
    ########################################################################################################

    ########################################################################################################
    # evaluate predictions from mcmc and svi and pbp
    ########################################################################################################
    pred_a = pred_mcmc  # reference model
    pred_b = [pred_svi, pred_pbp]  # list of models to compare with reference model
    plot_folder = [plot_folder1, plot_folder2]  # list of folders where the plots are saved
    model_a_names = 'MCMC'  # name of reference model
    model_b_names = ['SVI', 'PBP']  # list of model names

    for idx, preds in enumerate(pred_b):
        # use different plot folder for each model
        plot_settings.plot_folder = plot_folder[idx]
        region_ident1.gif_settings.path = plot_folder[idx]

        # set model names
        model_names = (model_a_names, model_b_names[idx])

        # set the models and get the wasserstein distance between the predictions
        model_evaluator1 = ModelEvaluator(
            predictions_a=pred_a, predictions_b=preds,
            wasserstein_distance=ws_dist_settings1, test_data=test_data)
        model_evaluator1.calc_wasserstein_distance()

        # calculate candidate regions and plot gif of critical distance
        model_evaluator1.calc_canidate_regions(region_ident=region_ident1)

        # calculate statistical tests (ANEES and binomial test)
        model_evaluator1.calc_statistical_tests(
            stat_test_settings=stat_test_settings, )

        # print results to console
        model_evaluator1.print_statistical_tests(model_names=model_names)

        # plot results to files
        model_evaluator1.plot_statistical_tests(
            plot_settings=plot_settings,
            ground_truth=ground_truth_distribution,
            model_names=model_names)

    ########################################################################################################


if __name__ == "__main__":
    # test some functionality
    test_functionality()
