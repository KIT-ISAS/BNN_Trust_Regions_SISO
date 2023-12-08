

from dataclasses import dataclass, field
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
    z_order_errorbar = 0.9
    z_grid_vlines = 0.8
    alpha_grid = 0.15
    grid_line_color = 'grey'
    region_x_label = r'Region $k$'
    fmt = 'none'
    x_pos_binom_bar_factor = 2/3
    binom_bar_color = 'purple'
    binom_marker_color = 'red'
    binom_marker = 'x'
    binom_y_label = r'$\pi$'
    binom_errorbar_label = r'$\pi$ Bounds'  # +' in Region'
    binom_p0_label = r'$\pi_0=$'
    prop_range = [-0.05, 1.05]

    anees_label = r'ANEES is $\chi^2$'
    anees_label_notchi2 = r'ANEES is not $\chi^2$'
    annes_errorbar_label = 'ANEES Bounds'
    split_label = 'Region Split'
    out_of_scope_label = 'ANEES o.s.'
    anees_y_label = 'ANEES'

    x_pos_anees_bar_factor = 1/3
    anees_bar_color = 'orange'
    anees_marker_color = 'blue'
    anees_marker = 'x'
    anees_marker_out_of_scope = '^'
    # diamond marker for regions where ANEES is not chi2 distributed
    nees_is_not_chi2_marker = 'D'
    anees_range = [-0.05, 2.5]
    max_anees_factor = 0.95


@dataclass
class DistributionPlotSettings:
    mean_label: str = 'predicted mean'
    quantile_label: str = 'CI'
    mean_zorder: float = 2
    ci_area_color: str = 'lightblue'
    ci_area_opacity: float = 1
    ci_area_zorder: float = 1.9


@dataclass
class PlotSettings:

    # TODO default values?
    # confidence interval of predictions that should be plotted in [0, 1]
    confidence_interval: float = 0.95
    image_format: str = 'svg'  # suppported image format for matplotlib 'png', 'svg' or 'pdf'
    plot_name: str = 'model_a'
    plot_folder: str = '.'  # folder where plots should be saved

    x_label: str = r'$x$'
    y_label: str = r'$y$'

    # prediction plot settings
    prediction_plot_settings: DistributionPlotSettings = field(
        default_factory=DistributionPlotSettings)

    # ground truth plot settings
    ground_truth_plot_settings: DistributionPlotSettings = DistributionPlotSettings

    error_bar_plot_settings: ErrorbarPlotSettings = ErrorbarPlotSettings

    # candidate region plot settings
    region_opacity: float = 0.5
    area_color_over_est: str = 'aquamarine'
    edge_color_over_est: str = 'black'
    hatch_over_est: str = '\\'
    under_est_color: str = 'lightsalmon'
    edge_color_under_est: str = 'black'
    hatch_under_est: str = '//'


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
                                             ):

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
        self._plot_regions(ax=ax)

        assert isinstance(ax, plt.Axes)  # only for type hints
        ax.legend()
        ax.set_xlabel(self.plot_settings.x_label)
        ax.set_ylabel(self.plot_settings.y_label)

        plot_type = 'pred'
        plot_path = self._plot_path(plot_type)
        plt.savefig(plot_path)

    def plot_stats_per_region(self, ):

        fig, ax = plt.subplots(constrained_layout=True)

        self._plot_stat_error_bar(ax)

        plot_type = 'stats'
        plot_path = self._plot_path(plot_type)
        plt.savefig(plot_path)

    def plot_stats_and_predictions(self, data, predictions: typing.Union[np.ndarray, UnivariateGaussian]):

        fig, ax = plt.subplots(2, constrained_layout=True)

        self._plot_stat_error_bar(ax[1])
        self._plot_predictions(predictions=predictions,
                               data=data,
                               ax=ax[0],
                               distribution_plot_settings=self.plot_settings.ground_truth_plot_settings)

        plot_type = 'stat_gt'
        plot_path = self._plot_path(plot_type)
        plt.savefig(plot_path)

    def plot_stats_and_ground_truth_dist(self, data, dist_to_ground_truth: np.ndarray, ):

        fig, ax = plt.subplots(2, constrained_layout=True)

        self._plot_stat_error_bar(ax[1])
        self._plot_predictions(predictions=ground_truth,
                               data=data,
                               ax=ax[0],
                               distribution_plot_settings=self.plot_settings.ground_truth_plot_settings)

        plot_type = 'stat_gt'
        plot_path = self._plot_path(plot_type)
        plt.savefig(plot_path)

    # function which returns ax instance with predictions

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
                zorder=plt_settings.mean_zorder)
        if quantile_predictions is None:
            return ax

        # plot quantiles as filled area
        ax.fill_between(data.input[:, 0], quantile_predictions[0, :], quantile_predictions[1, :],
                        color=plt_settings.ci_area_color,
                        alpha=plt_settings.ci_area_opacity,
                        label=plt_settings.quantile_label, zorder=plt_settings.ci_area_zorder)
        return ax

    def _plot_regions(self, ax: plt.Axes):
        """ Plot candidate regions and their statistical test results.

        :param ax: axis instance
        :type ax: plt.Axes
        :return: axis instance with candidate regions and their statistical test results
        :rtype: plt.Axes
        """

        plt_settings = self.plot_settings

        # plot candidate regions
        for region in self.candidate_regions.regions:
            # fill between x values
            uncertainty_over_est = region.anees_test_result.over_estimation
            uncertainty_under_est = region.anees_test_result.under_estimation
            if uncertainty_over_est:
                ax.axvspan(region.x_min, region.x_max, alpha=plt_settings.region_opacity,
                           color=plt_settings.area_color_over_est,
                           edgecolor=plt_settings.edge_color_over_est, hatch=plt_settings.hatch_over_est)
            elif uncertainty_under_est:
                ax.axvspan(region.x_min, region.x_max, alpha=plt_settings.region_opacity,
                           color=plt_settings.under_est_color,
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

        ax.set_xlim(self.candidate_regions.regions[0].x_min,
                    self.candidate_regions.regions[-1].x_max)

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

        # ax.legend()
        # ax2.legend()

        # added these three lines
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

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
        return os.path.join(self.plot_settings.plot_folder, self.plot_settings.plot_name + '_' +
                            plot_type + '.' + self.plot_settings.image_format)
