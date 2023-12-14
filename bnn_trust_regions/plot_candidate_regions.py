""" Module for plotting the candidate regions and their statistical test results of a SISO model."""

from dataclasses import dataclass, field
import math
import os
import typing

from matplotlib import pyplot as plt
import numpy as np

from .io_data import IOData
from .canidate_region import CandidateRegions
from .gaussian import UnivariateGaussian
from .utils.ci_prediction import calc_mean_and_quantiles


# TODO matplotlib settings should be set in a separate file
plt.rcParams.update({'errorbar.capsize': 4,
                     })


@dataclass
class ErrorbarPlotSettings:
    """
    Class representing the settings for error bar plots.
    """
    z_order_errorbar: float = 0.9
    z_grid_vlines: float = 0.8
    alpha_grid: float = 0.15
    grid_line_color: str = 'grey'
    region_x_label: str = r'Region $k$'
    fmt: str = 'none'
    x_pos_binom_bar_factor: float = 2/3
    binom_bar_color: str = 'purple'
    binom_marker_color: str = 'red'
    binom_marker: str = 'x'
    binom_y_label: str = r'$\pi$'
    binom_errorbar_label: str = r'$\pi$ Bounds'  # +' in Region'
    binom_p0_label: str = r'$\pi_0=$'
    prop_range: tuple = (-0.05, 1.05)

    anees_label: str = r'ANEES is $\chi^2$'
    anees_label_notchi2: str = r'ANEES is not $\chi^2$'
    annes_errorbar_label: str = 'ANEES Bounds'
    split_label: str = 'Region Split'
    out_of_scope_label: str = 'ANEES o.s.'
    anees_y_label: str = 'ANEES'

    x_pos_anees_bar_factor: float = 1/3
    anees_bar_color: str = 'tab:orange'
    anees_marker_color: str = 'tab:blue'
    anees_marker: str = 'x'
    anees_marker_out_of_scope: str = '^'
    # diamond marker for regions where ANEES is not chi2 distributed
    nees_is_not_chi2_marker: str = 'D'
    anees_range: tuple = (-0.05, 2.5)
    max_anees_factor: float = 0.95


@dataclass
class DistributionPlotSettings:
    """
    Class representing the settings for distribution plots.
    """
    mean_label: str = 'predicted mean'
    quantile_label: str = 'CI'
    mean_zorder: float = 2
    mean_color: str = 'tab:blue'
    mean_linestyle: str = '-'
    ci_area_color: str = 'lightblue'
    ci_area_opacity: float = 1
    ci_area_zorder: float = 1.9


@dataclass
class PlotSettings:
    """ Class representing the settings for plots."""
    # confidence interval of predictions that should be plotted in [0, 1]
    confidence_interval: float = 0.95
    image_format: str = 'svg'  # suppported image format for matplotlib 'png', 'svg' or 'pdf'
    model_name: str = 'model'
    plot_folder: str = '.'  # folder where plots should be saved

    first_ax_to_second_ax_ratio: float = 0.5

    x_label: str = r'$x$'
    y_label: str = r'$y$'
    wasserstein_label: str = r'$W_1$'

    # prediction plot settings
    prediction_plot_settings: DistributionPlotSettings = field(
        default_factory=DistributionPlotSettings)

    # ground truth plot settings
    ground_truth_plot_settings: DistributionPlotSettings = DistributionPlotSettings

    # wasserstein plot settings
    wasserstein_plot_settings: DistributionPlotSettings = DistributionPlotSettings

    error_bar_plot_settings: ErrorbarPlotSettings = ErrorbarPlotSettings

    # candidate region plot settings
    region_opacity: float = 0.5
    area_color_over_est: str = 'aquamarine'
    edge_color_over_est: str = 'black'
    hatch_over_est: str = '\\'
    under_est_color: str = 'lightsalmon'
    edge_color_under_est: str = 'black'
    hatch_under_est: str = '//'
    stats_subfolder_name: str = 'stats'

    def __post_init__(self):

        # create folder if it does not exist
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    # call method if plot folder is changed
    def __setattr__(self, name, value):
        if name == 'plot_folder':
            assert isinstance(value, str)
            # if path does not end with region_ident, add region_ident

            if not value.endswith(self.stats_subfolder_name):
                value = os.path.join(value, self.stats_subfolder_name)
            # Set plot folder for stats and create plot folder if it does not exist.
            if not os.path.exists(value):
                os.makedirs(value)
        super().__setattr__(name, value)


@dataclass
class PlotSisoCandidateRegions:
    """Class for plotting the candidate regions and their statistical test results of a SISO model.

    # TODO: add docstring
    """

    candidate_regions: CandidateRegions
    plot_settings: PlotSettings

    def __init__(self,
                 candidate_regions: CandidateRegions,
                 plot_settings: PlotSettings):

        self.candidate_regions = candidate_regions
        self.plot_settings = plot_settings

    def plot_predictions_with_region_results(self,
                                             predictions: typing.Union[np.ndarray, UnivariateGaussian],
                                             data: IOData,
                                             ground_truth: typing.Union[np.ndarray,
                                                                        UnivariateGaussian] = None,
                                             test_type: str = 'anees',
                                             ):
        """ Plot predictions and candidate regions with their statistical test results.

        :param predictions: predictions of a model
        :type predictions: typing.Union[np.ndarray, UnivariateGaussian]
        :param data: data used for predictions
        :type data: IOData
        :param ground_truth: ground truth distribution
        :type ground_truth: typing.Union[np.ndarray, UnivariateGaussian]
        :param test_type: type of statistical test, either 'anees' or 'binom'
        :type test_type: str
        """

        fig, ax = plt.subplots(constrained_layout=True)
        self._plot_predictions(predictions=predictions,
                               data=data,
                               ax=ax,
                               distribution_plot_settings=self.plot_settings.prediction_plot_settings)
        if ground_truth is not None:
            self._plot_predictions(predictions=ground_truth,
                                   data=data,
                                   ax=ax,
                                   distribution_plot_settings=self.plot_settings.ground_truth_plot_settings)

        self._plot_regions(ax=ax, test_type=test_type)

        assert isinstance(ax, plt.Axes)  # only for type hints
        ax.legend()
        ax.set_xlabel(self.plot_settings.x_label)
        ax.set_ylabel(self.plot_settings.y_label)

        plot_type = 'pred'
        plot_path = self._plot_path(plot_type)
        fig.savefig(plot_path)
        plt.close(fig)

    def plot_stats_per_region(self, ):
        """
        The function plots results of the statistical tests per input region and saves the plot as an image.
        """

        fig, ax = plt.subplots(constrained_layout=True)

        # set x axis limits to min and max of data
        ax.set_xlim(self.candidate_regions.regions[0].x_min,
                    self.candidate_regions.regions[-1].x_max)
        self._plot_stat_error_bar(ax)

        plot_type = 'stats'
        plot_path = self._plot_path(plot_type)
        fig.savefig(plot_path)

        self._save_legend_as_separate_image(fig, plot_type)
        plt.close(fig)

    def plot_stats_and_predictions(self, data, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                                   test_type: str = 'anees',):
        """
        Plot statistical test results and predictions.

        :param data: data used for predictions
        :type data: IOData
        :param predictions: predictions of a model
        :type predictions: typing.Union[np.ndarray, UnivariateGaussian]
        :param test_type: type of statistical test, either 'anees' or 'binom'
        :type test_type: str
        """

        fig, ax = plt.subplots(
            2, constrained_layout=True, sharex=True,
            gridspec_kw={'height_ratios': [self.plot_settings.first_ax_to_second_ax_ratio, 1.]})

        # set x axis limits to min and max of data
        ax[0].set_xlim(self.candidate_regions.regions[0].x_min,
                       self.candidate_regions.regions[-1].x_max)

        # ax[0] and ax[1] should have the same x axis scaling
        ax[1].set_xlim(ax[0].get_xlim())

        self._plot_stat_error_bar(ax[1])
        self._plot_predictions(predictions=predictions,
                               data=data,
                               ax=ax[0],
                               distribution_plot_settings=self.plot_settings.ground_truth_plot_settings)

        self._plot_regions(ax=ax[0], test_type=test_type)

        ax[0].set_xlabel(self.plot_settings.x_label)
        ax[0].set_ylabel(self.plot_settings.y_label)

        plot_type = 'stat_pred'
        plot_path = self._plot_path(plot_type)
        fig.savefig(plot_path)

        self._save_legend_as_separate_image(fig, plot_type)
        plt.close(fig)

    def plot_stats_and_ground_truth_dist(self, data, dist_to_ground_truth: np.ndarray,
                                         test_type: str = 'anees',):
        """ 
        Plot statistical test results and ground truth distribution.

        :param data: data used for predictions
        :type data: IOData
        :param dist_to_ground_truth: distance to ground truth distribution
        :type dist_to_ground_truth: np.ndarray
        :param test_type: type of statistical test, either 'anees' or 'binom'
        :type test_type: str


        """

        # set ratio of first axis to second axis
        fig, ax = plt.subplots(
            2, constrained_layout=True, sharex=True,
            gridspec_kw={'height_ratios': [self.plot_settings.first_ax_to_second_ax_ratio, 1.]})

        # set x axis limits to min and max of data
        ax[0].set_xlim(self.candidate_regions.regions[0].x_min,
                       self.candidate_regions.regions[-1].x_max)

        # ax[0] and ax[1] should have the same x axis scaling
        ax[1].set_xlim(ax[0].get_xlim())

        ax[0].set_xlabel(self.plot_settings.x_label)
        ax[0].set_ylabel(self.plot_settings.wasserstein_label)

        self._plot_stat_error_bar(ax[1])
        self._plot_predictions(predictions=dist_to_ground_truth,
                               data=data,
                               ax=ax[0],
                               distribution_plot_settings=self.plot_settings.wasserstein_plot_settings)

        # plot test results as colered areas
        self._plot_regions(ax=ax[0], test_type=test_type)

        plot_type = 'stat_gt'
        plot_path = self._plot_path(plot_type)
        fig.savefig(plot_path)
        self._save_legend_as_separate_image(fig, plot_type)
        plt.close(fig)

    def _get_legend_of_different_axes(self, ax_list: typing.List[plt.Axes], ):
        """ Combine legend of different axes. 
        :param ax: list of axis instances
        :type ax: typing.List[plt.Axes]
        """

        if not isinstance(ax_list, list):
            handles, labels = ax_list.get_legend_handles_labels()
            return handles, labels

        handles_list = []
        labels_list = []
        for ax_ in ax_list:
            # if ax contains list of axes instances, loop over them
            if isinstance(ax_, list):
                handles, labels = self._get_legend_of_different_axes(ax_)
                handles_list = handles_list + handles
                labels_list = labels_list + labels
                continue

            # get handles and labels
            handles, labels = ax_.get_legend_handles_labels()
            handles_list = handles_list + handles
            labels_list = labels_list + labels

        return handles_list, labels_list

    def _combine_legend_of_different_axes(self, fig: plt.Figure, ):

        ax_list = fig.get_axes()

        handles_list, labels_list = self._get_legend_of_different_axes(ax_list)
        # specify default order of items in legend
        order = range(len(handles_list))
        # switch last and second last item in order
        order = list(order)

        return handles_list, labels_list, order

    def _save_legend_as_separate_image(self, fig: plt.Figure, plot_type: str):

        plot_type += '_legend'
        plot_path = self._plot_path(plot_type)

        # get handles and labels
        handles_list, labels_list, order = self._combine_legend_of_different_axes(fig)

        # change order of legend items
        order[-1], order[-2] = order[-2], order[-1]

        # add legend to plot
        lgd = fig.legend([handles_list[idx] for idx in order], [labels_list[idx]
                                                                for idx in order],
                         loc='upper right', bbox_to_anchor=(10, 0),
                         ncols=math.ceil(len(labels_list)/2))

        fig.savefig(plot_path,
                    bbox_inches=lgd.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),)
        plt.close(fig)

    def _plot_predictions(self, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                          data: IOData,
                          ax: plt.Axes,
                          distribution_plot_settings: DistributionPlotSettings,
                          ):
        """ Plot predictions of a model.

        :param predictions: predictions of a model
        :type predictions: typing.Union[np.ndarray, UnivariateGaussian]
        :param data: data used for predictions
        :type data: IOData
        :param confidence_interval: confidence interval of predictions that should be plotted in [0, 1]
        :type confidence_interval: float
        :param ax: axis instance
        :type ax: plt.Axes
        :return: axis instance with predictions
        :rtype: plt.Axes
        """

        plt_settings = distribution_plot_settings

        mean_predictions, quantile_predictions = calc_mean_and_quantiles(
            predictions, self.plot_settings.confidence_interval)

        ax.plot(data.input, mean_predictions, label=plt_settings.mean_label,
                color=plt_settings.mean_color, linestyle=plt_settings.mean_linestyle,
                zorder=plt_settings.mean_zorder)
        if quantile_predictions is None:
            return ax

        ci_label = f'{100*self.plot_settings.confidence_interval:0.1f}' + r' $\%$ CI'

        # plot quantiles as filled area
        ax.fill_between(data.input[:, 0], quantile_predictions[0, :], quantile_predictions[1, :],
                        color=plt_settings.ci_area_color,
                        alpha=plt_settings.ci_area_opacity,
                        label=ci_label, zorder=plt_settings.ci_area_zorder)
        return ax

    def _plot_regions(self, ax: plt.Axes, test_type: str = 'anees'):
        """ Plot candidate regions and their statistical test results.

        :param ax: axis instance
        :type ax: plt.Axes
        :param test_type: type of statistical test, either 'anees' or 'binom'
        :type test_type: str

        :return: axis instance with candidate regions and their statistical test results
        :rtype: plt.Axes
        """

        plt_settings = self.plot_settings

        # plot candidate regions
        for region in self.candidate_regions.regions:
            # fill between x values
            if test_type == 'anees':
                uncertainty_over_est = region.anees_test_result.over_estimation
                uncertainty_under_est = region.anees_test_result.under_estimation
            elif test_type == 'binom':
                uncertainty_over_est = region.binom_test_result.over_estimation
                uncertainty_under_est = region.binom_test_result.under_estimation
            else:
                raise ValueError("test__type must be either 'anees' or 'binom'")

            if uncertainty_over_est:
                ax.axvspan(region.x_min, region.x_max, alpha=plt_settings.region_opacity,
                           facecolor=plt_settings.area_color_over_est,
                           edgecolor=plt_settings.edge_color_over_est, hatch=plt_settings.hatch_over_est)
            elif uncertainty_under_est:
                ax.axvspan(region.x_min, region.x_max, alpha=plt_settings.region_opacity,
                           facecolor=plt_settings.under_est_color,
                           edgecolor=plt_settings.edge_color_under_est, hatch=plt_settings.hatch_under_est)
            else:
                # calibrated region
                # anything to visualize?
                pass

        return ax

    def _plot_stat_error_bar(self, ax: plt.Axes):
        """ Plot statistical test results per region.

        :return: axis instance with statistical test results per region
        :rtype: plt.Axes
        """

        plt_settings = self.plot_settings.error_bar_plot_settings

        ax2 = ax.twinx()

        x_pos_bar_binom, x_pos_bar_anees, x_label_position, x_region_split_points = \
            self._get_x_positons_for_errorbars(
                plt_settings.x_pos_binom_bar_factor, plt_settings.x_pos_anees_bar_factor)
        binom_stat, binom_bounds, tested_proportion, _ = self.candidate_regions.get_binom_results()
        anees_stat, anees_crit_deviations_from_1, nees_is_chi2, _ = self.candidate_regions.get_anees_results()

        binom_p0_label = plt_settings.binom_p0_label + r'$' + f'{tested_proportion:0.2f}' + r'$'

        # anees test results
        max_anees = plt_settings.max_anees_factor * plt_settings.anees_range[1]
        anees_greater_max = anees_stat > max_anees  # plot with different markers when value is out of scope
        anees_lower_max_and_chi2 = np.logical_and(anees_stat <= max_anees, nees_is_chi2)
        anees_lower_max_and_not_chi2 = np.logical_and(
            anees_stat <= max_anees, np.logical_not(nees_is_chi2))

        ax.errorbar(x=x_pos_bar_anees,
                    y=np.ones_like(x_pos_bar_anees),  # perfect calibration is ANEES=1
                    yerr=anees_crit_deviations_from_1,
                    label=plt_settings.annes_errorbar_label,
                    # capsize=plt_settings.errorbar_capsize,  # set capsize
                    fmt=plt_settings.fmt, color=plt_settings.anees_bar_color, zorder=plt_settings.z_order_errorbar)

        # plot anees if its lower than max_anees and is chi2 distributed
        ax.plot(x_pos_bar_anees[anees_lower_max_and_chi2], anees_stat[anees_lower_max_and_chi2],
                marker=plt_settings.anees_marker, zorder=plt_settings.z_order_errorbar,
                color=plt_settings.anees_marker_color, label=plt_settings.anees_label,
                linestyle='None')
        # plot anees if its lower than max_anees and is not chi2 distributed
        ax.plot(x_pos_bar_anees[anees_lower_max_and_not_chi2], anees_stat[anees_lower_max_and_not_chi2],
                marker=plt_settings.nees_is_not_chi2_marker, zorder=plt_settings.z_order_errorbar,
                color=plt_settings.anees_marker_color, label=plt_settings.anees_label_notchi2,
                linestyle='None')
        # number of anees values > max as integer
        num_anees_greater_max = np.sum(anees_greater_max)
        # plot anees if its greater than max_anees
        ax.plot(x_pos_bar_anees[anees_greater_max], max_anees * np.ones((num_anees_greater_max,)),
                marker=plt_settings.anees_marker_out_of_scope, zorder=plt_settings.z_order_errorbar,
                color=plt_settings.anees_marker_color, label=plt_settings.out_of_scope_label,
                linestyle='None')

        # binomial test results
        assert isinstance(ax, plt.Axes)  # only for type hints
        ax2.errorbar(x=x_pos_bar_binom,
                     y=binom_stat,
                     yerr=binom_bounds,
                     label=plt_settings.binom_errorbar_label,
                     #  capsize=plt_settings.errorbar_capsize,  # set capsize
                     fmt=plt_settings.fmt, color=plt_settings.binom_bar_color, zorder=plt_settings.z_order_errorbar)

        ax2.plot(x_pos_bar_binom, tested_proportion * np.ones_like(x_pos_bar_binom),
                 label=binom_p0_label,
                 marker=plt_settings.binom_marker, zorder=plt_settings.z_order_errorbar,
                 color=plt_settings.binom_marker_color, linestyle='None')

        # grid lines for binomial errorbar and scatter points
        ax2.vlines(x_pos_bar_binom,
                   ymin=plt_settings.prop_range[0], ymax=plt_settings.prop_range[1],
                   color=plt_settings.grid_line_color, linestyles='dotted',
                   alpha=plt_settings.alpha_grid, zorder=plt_settings.z_grid_vlines)
        # grid lines for anees errorbars and scatter points
        ax2.vlines(x_pos_bar_anees,
                   ymin=plt_settings.prop_range[0], ymax=plt_settings.prop_range[1],
                   color=plt_settings.grid_line_color, linestyles='dotted',
                   alpha=plt_settings.alpha_grid, zorder=plt_settings.z_grid_vlines)

        # grid lines for regions
        # add vertical lines to mark regions
        ax2.vlines(x_region_split_points[1:-1],  # splitting points without first and last data points
                   ymin=plt_settings.prop_range[0], ymax=plt_settings.prop_range[1],
                   color=plt_settings.grid_line_color, linestyles='dashed',
                   label=plt_settings.split_label)

        x_label = [f'{x:n}' for x in range(len(x_label_position))]

        if len(x_label) >= 13:
            # ax.set_xticks(x_label_position, minor=True)
            # use every second label
            ax.set_xticks(x_label_position[::2], minor=False)
            ax.set_xticklabels(x_label[::2])
        else:
            ax.set_xticks(x_label_position)
            ax.set_xticklabels(x_label)

        ax.set_xlabel(plt_settings.region_x_label)
        ax.set_ylabel(plt_settings.anees_y_label)
        ax.set_ylim(plt_settings.anees_range)

        ax2.set_ylabel(plt_settings.binom_y_label)
        ax2.set_ylim(plt_settings.prop_range)

        # ax = [ax, ax2]
        return ax

        # plot marker for regions w

    def _get_x_positons_for_errorbars(self, x_pos_binom_bar_factor: float, x_pos_anees_bar_factor: float):
        # initialize array for x position of error bar
        x_pos_bar_binom = np.empty((self.candidate_regions.get_num_regions(), ))
        x_pos_bar_anees = np.empty_like(x_pos_bar_binom)
        x_label_position = np.empty_like(x_pos_bar_binom)
        # initialize array for x position of vertical line which split the regions
        # add one more element to array to plot vertical line at the end of the last region
        x_region_split_points = np.empty((self.candidate_regions.get_num_regions()+1, ))

        # plot stat confidence interval per region as error bar
        for idx, region in enumerate(self.candidate_regions.regions):

            max_min_diff = region.x_max - region.x_min

            # calculate x position of binom error bar
            x_pos_bar_binom[idx] = region.x_min + max_min_diff * x_pos_binom_bar_factor
            x_pos_bar_anees[idx] = region.x_min + max_min_diff * x_pos_anees_bar_factor
            x_label_position[idx] = region.x_min + max_min_diff * 0.5

            x_region_split_points[idx] = region.x_min

        # add last region x_max to array
        x_region_split_points[-1] = self.candidate_regions.regions[-1].x_max

        return x_pos_bar_binom, x_pos_bar_anees, x_label_position, x_region_split_points

    def _plot_path(self, plot_type: str):
        """ get path to save plot 
        :param plot_type: name which is used in the plot file name
        :type plot_type: str
        """
        return os.path.join(self.plot_settings.plot_folder, self.plot_settings.model_name + '_' +
                            plot_type + '.' + self.plot_settings.image_format)
