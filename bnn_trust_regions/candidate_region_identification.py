""" class CandidateRegionIdentification to identify candidate regions in the input space """

import copy
from dataclasses import dataclass
import logging
import os

import imageio
from matplotlib import pyplot as plt

import numpy as np
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

    def __init__(self,
                 raw_distances: np.ndarray,
                 test_data: IOData,
                 min_points_per_region: int = 200,
                 smoothing_window_size: int = 50,
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

    def calc_critical_distance(self, verbose: bool = False, gif_settings: IdentGifSettings = None):
        """
        Calculate the critical distance.

        :param verbose: Whether to create a gif of critical distance selection.
        :type verbose: bool, optional
        :param gif_settings: The gif settings.
        :type gif_settings: IdentGifSettings, optional
        """

        # if verbose, create gif of critical distance selection
        if verbose:
            frames = []
            if gif_settings is None:
                gif_settings = IdentGifSettings()

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
                frame = _create_frame(idx, self.test_data.input, crit_value, valid_invalid_switching,
                                      dist=distance, gif_settings=gif_settings)
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
                    frame = _create_frame(idx, self.test_data.input, crit_value, valid_invalid_switching,
                                          dist=distance, gif_settings=gif_settings)
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

    def get_candidate_regions(self):
        """
        The function returns the candidate regions.
        Critical distance defines rough candidate regions.
        Use the minimum number of points per region to get a finder subdevision.

        The candidate regions are defined by the switching points.

        :return: The candidate regions.
        :rtype: numpy.ndarray
        """
        return self.switching_points


def _create_frame(idx: int, input_data: np.ndarray, crit_value: float,
                  valid_invalid_switching: list, dist: np.ndarray, gif_settings: IdentGifSettings):
    logging.debug(
        f'Iteration: {idx}, crit value: {crit_value}, num switching: {len(valid_invalid_switching)}')
    ax = plt.gca()
    assert isinstance(ax, plt.Axes)
    ax.clear()
    ax.set(xlabel='x', ylabel='Wasserstein distance')
    plt.plot(input_data.squeeze(), dist, color='b')

    plt.axhline(y=crit_value, color='k', linestyle='--')
    # fill areas between switching points
    # the areas should be white if distance is lower than crit value
    # and red if distance is higher than crit value
    extended_switching = [0, *valid_invalid_switching, input_data.shape[0]-1]
    for i in range(0, len(extended_switching), 2):
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
    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')
    file_path = f'.tmp/wasserstein_dist_animation_{idx}.png'
    plt.savefig(file_path, dpi=gif_settings.dpi)
    return imageio.v3.imread(file_path)
