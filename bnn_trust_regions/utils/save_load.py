""" utils module"""

import os
import pickle
import typing

import numpy as np


def save_io_data(folder: str, file_name: str, input_data: np.ndarray, output_data: np.ndarray):
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump((input_data, output_data), f)


def load_io_data(folder: str, file_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        input_data, output_data = pickle.load(f)
    return input_data, output_data


def save_sampled_predictions(folder: str, file_name: str, sampled_predictions: np.ndarray,):
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump(sampled_predictions, f)


def load_sampled_predictions(folder: str, file_name: str) -> np.ndarray:
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        sampled_predictions = pickle.load(f)
    return sampled_predictions


def save_gaussian_predictions(folder: str, file_name: str, mean: typing.Union[np.ndarray, float], var: typing.Union[np.ndarray, float]):
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump((mean, var), f)


def load_gaussian_predictions(folder: str, file_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        mean, var = pickle.load(f)
    return mean, var
