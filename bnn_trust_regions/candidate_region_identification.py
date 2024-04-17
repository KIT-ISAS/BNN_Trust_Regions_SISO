""" class CandidateRegionIdentification to identify candidate regions in the input space """


import copy
from dataclasses import dataclass
import os
import typing

import imageio
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing
import scipy.signal

from .canidate_region import CandidateRegion, CandidateRegions
from .io_data import IOData
from .gaussian import UnivariateGaussian


@dataclass
class IdentGifSettings:
    """ Settings for creating a gif of the critical distance selection.
    :param path: The `path` parameter is a string that specifies the path where the gif should be saved.
    :type path: str, optional
    :param file_name: The `file_name` parameter is a string that specifies the name of the gif file.
    :type file_name: str, optional
    :param dpi: The `dpi` parameter is an integer that specifies the resolution of the gif.
    :type dpi: int, optional
    :param fps: The `fps` parameter is an integer that specifies the number of frames per second.
    :type fps: int, optional
    :param loop: The `loop` parameter is an integer that specifies the number of times the gif should loop.
    :type loop: int, optional"""
    path: str = '.'
    file_name: str = 'crit_dist.gif'
    dpi: int = 200
    fps: int = 2
    loop: int = 0  # 0 means infinite loop, 1 means no loop
    region_ident_subfolder_name: str = 'region_ident'

    # def __init__(self, path: str = '.', file_name: str = 'crit_dist.gif', dpi: int = 200, fps: int = 2, loop: int = 0):
    #     self.path = path
    #     self.file_name = file_name
    #     self.dpi = dpi
    #     self.fps = fps
    #     self.loop = loop

    # if path is changed, call method to create new folder
    def __setattr__(self, prop, value):
        if prop == 'path':
            value = self._check_path(value)
        super().__setattr__(prop, value)

    def _check_path(self, value: str,):
        """ Check if path exists. If not, create it."""
        assert isinstance(value, str)
        # if path does not end with region_ident, add region_ident
        if not value.endswith(self.region_ident_subfolder_name):
            value = os.path.join(value, self.region_ident_subfolder_name)
        # Set plot folder for stats and create plot folder if it does not exist.
        if not os.path.exists(value):
            os.makedirs(value)
        return value


@dataclass
class SisoCandidateRegionIdentification:
    """
    SISO candidate region identification class.
    """

    # used data
    raw_distances: np.ndarray
    test_data: IOData

    # hyperparameters
    smoothing_window_size: int
    min_points_per_region: int

    # only internal use?
    smoothed_distances: np.ndarray

    # identified candidate regions
    critical_distance: float
    switching_idxs: np.ndarray  # switching index of candeidate regions
    extendend_switching_idxs: np.ndarray  # same as switching_idxs but with first and last index added
    candidate_region_list: typing.List[CandidateRegion]

    # only for plotting
    gif_settings: IdentGifSettings
    verbose: bool = False

    def __init__(self,
                 raw_distances: typing.Union[np.ndarray, None] = None,
                 test_data: typing.Union[IOData, None] = None,
                 min_points_per_region: int = 200,
                 smoothing_window_size: int = 50,
                 verbose: bool = False,
                 gif_settings: typing.Union[IdentGifSettings, None] = None
                 ):
        """
        The function initializes the CandidateRegionIdentification class.

        :param raw_distances: The `raw_distances` parameter is a numpy array that contains the raw distances
        between the reference model and the approximation model.
        :type raw_distances: np.ndarray
        :param smoothing_window_size: The `smoothing_window_size` parameter is an integer that specifies the
        window size used for smoothing the raw distances. The default value is 5.
        :type smoothing_window_size: int (optional)
        :param critical_distance: The `critical_distance` parameter is a float that specifies the critical
        distance used for identifying candidate regions. The default value is 0.5.
        :type critical_distance: float (optional)
        """
        self.raw_distances = raw_distances
        self.smoothing_window_size = smoothing_window_size
        self.test_data = test_data
        self.min_points_per_region = min_points_per_region

        # if test_data is not None:
        #     self.num_distributions = test_data.output.shape[0]

        self.verbose = verbose
        self.gif_settings = gif_settings

    def smooth_distances(self):
        """
        The function smooths the raw distances using a moving average filter.

        :return: The smoothed distances.
        :rtype: numpy.ndarray
        """
        window_size = self.smoothing_window_size
        distances = copy.deepcopy(self.raw_distances)
        if window_size % 2 == 0:
            pad_width = (int((window_size-1) / 2), int((window_size-1) / 2) + 1)
        else:
            pad_width = int((window_size-1) / 2)
        padded = np.pad(
            distances, (pad_width,), mode='edge')
        self.smoothed_distances = scipy.signal.convolve(
            padded, np.ones((window_size,))/window_size, mode='valid')

    def calc_critical_distance(self, ):
        """
        Calculate the critical distance.

        :param verbose: Whether to create a gif of critical distance selection.
        :type verbose: bool, optional
        :param gif_settings: The gif settings.
        :type gif_settings: IdentGifSettings, optional
        """

        # if verbose, create gif of critical distance selection
        if self.verbose:
            frames = []
            if self.gif_settings is None:
                self.gif_settings = IdentGifSettings()
        gif_settings = self.gif_settings
        verbose = self.verbose

        distance = self.smoothed_distances
        sorted_distance = np.sort(self.smoothed_distances)
        crit_value = sorted_distance[0]

        last_num_slices = 0
        max_slices = 0

        required_min_points = self.min_points_per_region
        for idx, crit_value in enumerate(sorted_distance):

            lower_crit = np.array(distance <= crit_value)

            # get switching indices
            valid_invalid_switching = np.where(lower_crit[:-1] != lower_crit[1:])[0]

            # only for plots
            if verbose and (idx % 100 == 0):
                frame = _create_frame(self.test_data.input, crit_value, valid_invalid_switching,
                                      dist=distance, gif_settings=gif_settings, idx=idx, )
                frames.append(frame)

            # count switching frequency
            num_slices = valid_invalid_switching.shape[0] + 1
            splited_output = np.split(self.test_data.output, valid_invalid_switching, axis=0)
            # list of lengths of each region
            list_of_lengths = [len(item) for item in splited_output]

            min_points_per_region = min(
                list_of_lengths[1:-1]) if len(list_of_lengths) > 2 else min(list_of_lengths)
            max_slices = max(num_slices, max_slices)

            # logging.debug
            # if verbose:
            #     print(
            #         f'Number of slices: {num_slices}, last num slices: {last_num_slices}, max num slices: {max_slices}')

            # stopp increasing of critical value when following conditions are satisfied
            # min_num_cluster_condition = num_slices > 1  # use more than one cluster
            # min_points_per_cluster_condition = min_points_per_region \
            #     >= required_min_points  # min cluster size

            if ((num_slices > 1) and ((min_points_per_region >= required_min_points))):
                self.critical_distance = crit_value
                self.switching_idxs = valid_invalid_switching

                # final frame with chosen crit value
                if verbose:
                    frame = _create_frame(self.test_data.input, crit_value, valid_invalid_switching,
                                          dist=distance, gif_settings=gif_settings, idx=idx, )
                    frames.append(frame)
                    frames.append(frame)  # add last frame twice to make gif loop nicer
                    if gif_settings.path is None:
                        gif_settings.path = '.'  # save in current directory
                    file_path = os.path.join(gif_settings.path, f'{gif_settings.file_name}.gif')
                    imageio.mimsave(file_path, frames, format='GIF',
                                    fps=gif_settings.fps, loop=gif_settings.loop)

                return

            last_num_slices = num_slices
            _ = last_num_slices  # only for debugging?

        raise ValueError(
            'No critical value found. Please check if the data and predictions are sorted according to the input data.' +
            'If the data is sorted, try smaller min_points_per_region.')

    def subsplit_candidate_regions(self,):
        """
        The function `subsplit_candidate_regions` takes a range of indices and splits it into finer candidate regions of a minimum
        size, returning the indices that belong to the canidate regions and the full list of indices.

        """
        valid_invalid_switching = self.switching_idxs

        num_predictions = self.test_data.output.shape[0]
        # Add the bounds of the range to the list
        extended_switching_range = [0, *valid_invalid_switching, num_predictions - 1]
        # remove duplicates, if first or last index is already in list
        extended_switching_range = list(dict.fromkeys(extended_switching_range))

        # Calculate the sizes of each cluster in the range
        sizes_of_clusters = np.diff(extended_switching_range)

        # Create a copy of the range to add new indices to
        new_extended_switching_range = copy.deepcopy(extended_switching_range)

        # Calculate the number of subclusters to create for each cluster
        num_create_split_points = np.floor(
            sizes_of_clusters / self.min_points_per_region).astype(int)-1

        # Add new indices to the range for each cluster that needs to be split
        adapted_index = 0
        for index, num_add_cluster in enumerate(num_create_split_points):
            if num_add_cluster > 0:
                start_index = extended_switching_range[index]

                # Calculate the indices of the new subclusters
                add_indices = start_index \
                    + np.ceil((sizes_of_clusters[index] / (num_add_cluster+1)) *  # index scaling factor
                              np.arange(start=1, stop=num_add_cluster+1, step=1)).astype(int)  # interval [1, num_add_cluster]

                # Insert the new subclusters into the range
                new_extended_switching_range = np.insert(
                    new_extended_switching_range, index+adapted_index+1, add_indices)
                adapted_index = adapted_index + num_add_cluster  # increase counter

        # Remove the bounds of the range from the list
        valid_invalid_switching = new_extended_switching_range[1:-1]
        self.switching_idxs = valid_invalid_switching
        self.extendend_switching_idxs = new_extended_switching_range

        if self.verbose:
            _create_frame(self.test_data.input, self.critical_distance, valid_invalid_switching,
                          dist=self.smoothed_distances, gif_settings=self.gif_settings, idx='final')

    def split_data_in_regions(self, predictions: typing.Union[np.ndarray, UnivariateGaussian],
                              ) -> CandidateRegions:
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

        # num_predictions = self.num_distributions
        output_data = self.test_data.output

        switching_idxs = self.switching_idxs
        extendend_switching_idxs = self.extendend_switching_idxs

        # get input values which define the regions
        region_bounds = self.test_data.input[extendend_switching_idxs]
        candidate_region_list = []

        # if all prediction are valid or invalid -> dont split
        if 0 < switching_idxs.size < output_data.size:

            splited_outputs = np.split(output_data, switching_idxs, axis=0)
            if isinstance(predictions, UnivariateGaussian):

                splitted_means = np.split(predictions.mean, switching_idxs, axis=0)
                splitted_vars = np.split(predictions.var, switching_idxs, axis=0)
                splited_predictions = []
                for idx, (mean, var, splited_output) in enumerate(zip(splitted_means, splitted_vars, splited_outputs)):
                    predictions_in_region = UnivariateGaussian(mean=mean, var=var)
                    # splited_predictions.append(predictions_in_region)

                    candidate_region = CandidateRegion(predictions_in_region=predictions_in_region,
                                                       outputs_in_region=splited_output,
                                                       x_min=region_bounds[idx], x_max=region_bounds[idx+1],)
                    candidate_region_list.append(candidate_region)

            else:
                splited_predictions = np.split(predictions, switching_idxs, axis=1)

                for idx, (splitted_prediction, splited_output) in enumerate(zip(splited_predictions, splited_outputs)):
                    candidate_region = CandidateRegion(
                        predictions_in_region=splitted_prediction,
                        outputs_in_region=splited_output,
                        x_min=region_bounds[idx],
                        x_max=region_bounds[idx+1],)
                    candidate_region_list.append(candidate_region)
        else:
            candidate_region = CandidateRegion(predictions_in_region=predictions,
                                               outputs_in_region=output_data,
                                               x_min=region_bounds[0],
                                               x_max=region_bounds[-1],)
            candidate_region_list.append(candidate_region)

        self.candidate_region_list = candidate_region_list
        # return list of candidate regions
        return CandidateRegions(candidate_region_list)


def _create_frame(input_data: np.ndarray, crit_value: float,
                  valid_invalid_switching: list, dist: np.ndarray,
                  gif_settings: IdentGifSettings, idx: typing.Union[int, str, None] = None, initial=False) -> numpy.typing.ArrayLike:
    """
    The `_create_frame` function creates a frame for an animation by plotting data and saving it as an
    image.

    :param input_data: The `input_data` parameter is a numpy array containing the input data for the
    plot. It represents the x-axis values of the plot
    :type input_data: np.ndarray
    :param crit_value: The `crit_value` parameter is a float value that represents the critical value
    for the Wasserstein distance. It is used to determine whether the distance between two points is
    considered valid or invalid
    :type crit_value: float
    :param valid_invalid_switching: The parameter `valid_invalid_switching` is a list that contains the
    indices of the switching points in the `input_data` array. These switching points represent the
    points where the data transitions from being valid to invalid or vice versa
    :type valid_invalid_switching: list
    :param dist: `dist` is a numpy array representing the Wasserstein distance values. It is used to
    plot the Wasserstein distance against the input data
    :type dist: np.ndarray
    :param gif_settings: The `gif_settings` parameter is an instance of the `IdentGifSettings` class. It
    contains settings for creating the GIF animation
    :type gif_settings: IdentGifSettings
    :param idx: The `idx` parameter is an optional parameter that specifies the iteration index. If it
    is not provided, the value `'final'` is used as the default value
    :type idx: int
    :param initial: The `initial` parameter is a boolean value that specifies whether the frame is the
    initial frame. If it is set to True, the frame is the initial frame. Otherwise, it is set to False.
    The intial frame contains the initial plot without any switching points (only distance values over x)
    :type initial: bool
    :return: an image file in PNG format.
    :rtype: numpy.typing.ArrayLike
    """
    if idx is None:
        idx = 'final'
    elif isinstance(idx, int):
        # shift idx by 1 to start at 1
        # idx 0 is the initial frame
        idx += 1

        # if initial frame, plot only distance values at idx 0
        if idx == 1:
            initial = True
            _create_frame(input_data, crit_value, valid_invalid_switching, dist, gif_settings, -1, initial)

    # if idx not None or int, idx is used as str for file name

    # logging.debug
    # print(
    #     f'Iteration: {idx}, crit value: {crit_value}, num switching: {len(valid_invalid_switching)}')
    fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    # ax.clear()
    ax.set(xlabel=r'$x$', ylabel='Wasserstein distance')
    ax.plot(input_data.squeeze(), dist, color='b')

    ax.axhline(y=crit_value, color='k', linestyle='--')

    if not initial:
        # plot vertical lines at switching points
        for switching_point in valid_invalid_switching:
            ax.axvline(x=input_data.squeeze()[switching_point], color='k', linestyle='-.')
        # fill areas between switching points
        # the areas should be white if distance is lower than crit value
        # and red if distance is higher than crit value
        extended_switching = [0, *valid_invalid_switching, input_data.shape[0]-1]
        for i in range(0, len(extended_switching)-1):
            range_start = extended_switching[i]
            range_end = extended_switching[i+1]
            mean_raw_distance = np.mean(dist[range_start:range_end])
            if mean_raw_distance > crit_value:
                facecolor = 'r'
            else:
                facecolor = 'w'
            ax.axvspan(input_data.squeeze()[range_start], input_data.squeeze()[
                range_end], facecolor=facecolor, alpha=0.5)

    # make dir if .tmp does not exist
    if not os.path.exists(gif_settings.path):
        os.makedirs(gif_settings.path)

    file_path = os.path.join(gif_settings.path, f'wasserstein_dist_animation_{idx}.png')
    fig.savefig(file_path, dpi=gif_settings.dpi)
    plt.close()
    return imageio.v3.imread(file_path)
