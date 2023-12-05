""" class CandidateRegionIdentification to identify candidate regions in the input space """

import copy
from dataclasses import dataclass
import logging
import os

import imageio
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing
import scipy.signal

from .io_data import IOData


@dataclass
class IdentGifSettings:
    path: str = None
    file_name: str = 'crit_dist.gif'
    dpi: int = 200
    fps: int = 2
    loop: int = 0  # 0 means infinite loop, 1 means no loop


@dataclass
class CandidateRegionIdentification:
    raw_distances: np.ndarray
    smoothed_distances: np.ndarray
    smoothing_window_size: int

    test_data: IOData

    min_points_per_region: int

    critical_distance: float
    switching_points: np.ndarray
    extendend_switching_points: np.ndarray  # same as switching_points but with first and last index added

    # only for plotting
    gif_settings: IdentGifSettings
    verbose: bool = False

    def __init__(self,
                 raw_distances: np.ndarray = None,
                 test_data: IOData = None,
                 min_points_per_region: int = 200,
                 smoothing_window_size: int = 50,
                 verbose: bool = False,
                 gif_settings: IdentGifSettings = None
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

            logging.debug(
                f'Number of slices: {num_slices}, last num slices: {last_num_slices}, max num slices: {max_slices}')

            # stopp increasing of critical value when following conditions are satisfied
            # min_num_cluster_condition = num_slices > 1  # use more than one cluster
            # min_points_per_cluster_condition = min_points_per_region \
            #     >= required_min_points  # min cluster size

            if ((num_slices > 1) and ((min_points_per_region >= required_min_points))):
                self.critical_distance = crit_value
                self.switching_points = valid_invalid_switching

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

        raise ValueError(
            'No critical value found. Please check the input data.')

    # def get_candidate_regions(self):
    #     """
    #     The function returns the candidate regions.
    #     Critical distance defines rough candidate regions.
    #     Use the minimum number of points per region to get a finder subdevision.

    #     The candidate regions are defined by the switching points.

    #     :return: The candidate regions.
    #     :rtype: numpy.ndarray
    #     """

    #     splited_prediction, splited_output, extended_switching_range = split_in_local_clusters(
    #         prediction=prediction, output_data=output_data, invalid_range=invalid_range, min_points_per_cluster=min_points_per_cluster)

    #     return self.switching_points

    def subsplit_candidate_regions(self,):
        """
        The function `subsplit_candidate_regions` takes a range of indices and splits it into finer candidate regions of a minimum
        size, returning the indices that belong to the canidate regions and the full list of indices.

        """
        valid_invalid_switching = self.switching_points

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
        self.switching_points = valid_invalid_switching
        self.extendend_switching_points = new_extended_switching_range

        _create_frame(self.test_data.input, self.critical_distance, valid_invalid_switching,
                      dist=self.smoothed_distances, gif_settings=self.gif_settings, idx='final')


def _create_frame(input_data: np.ndarray, crit_value: float,
                  valid_invalid_switching: list, dist: np.ndarray,
                  gif_settings: IdentGifSettings, idx: int = None) -> numpy.typing.ArrayLike:
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
    :return: an image file in PNG format.
    :rtype: numpy.typing.ArrayLike
    """
    if idx is None:
        idx = 'final'
    logging.debug(
        f'Iteration: {idx}, crit value: {crit_value}, num switching: {len(valid_invalid_switching)}')
    ax = plt.gca()
    assert isinstance(ax, plt.Axes)
    ax.clear()
    ax.set(xlabel='x', ylabel='Wasserstein distance')
    plt.plot(input_data.squeeze(), dist, color='b')

    plt.axhline(y=crit_value, color='k', linestyle='--')

    # plot vertical lines at switching points
    for switching_point in valid_invalid_switching:
        plt.axvline(x=input_data.squeeze()[switching_point], color='k', linestyle='-.')
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
        plt.axvspan(input_data.squeeze()[range_start], input_data.squeeze()[
            range_end], facecolor=facecolor, alpha=0.5)
    # make dir if .tmp does not exist
    if not os.path.exists(gif_settings.path):
        os.makedirs(gif_settings.path)

    file_path = os.path.join(gif_settings.path, f'wasserstein_dist_animation_{idx}.png')
    plt.savefig(file_path, dpi=gif_settings.dpi)
    return imageio.v3.imread(file_path)
