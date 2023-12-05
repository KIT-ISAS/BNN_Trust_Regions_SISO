""" utils module"""

import os
import pickle
import typing

import numpy as np


def save_io_data(folder: str, file_name: str, input_data: np.ndarray, output_data: np.ndarray):
    """
    The function `save_io_data` saves input and output data as a pickle file in a specified folder.

    :param folder: The `folder` parameter is a string that represents the directory where you want to
    save the data
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file you
    want to save
    :type file_name: str
    :param input_data: The input_data parameter is a numpy array that contains the input data for your
    model. It could be a set of features or any other data that your model needs to make predictions
    :type input_data: np.ndarray
    :param output_data: The `output_data` parameter is a NumPy array that contains the output data that
    you want to save
    :type output_data: np.ndarray
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump((input_data, output_data), f)


def load_io_data(folder: str, file_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    The function `load_io_data` loads input and output data from a pickle file in a specified folder.

    :param folder: The `folder` parameter is a string that represents the path to the folder where the
    data file is located
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file you
    want to load
    :type file_name: str
    :return: The function load_io_data returns a tuple containing two numpy arrays: input_data and
    output_data.
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        input_data, output_data = pickle.load(f)
    return input_data, output_data


def save_sampled_predictions(folder: str, file_name: str, sampled_predictions: np.ndarray,):
    """
    The function `save_sampled_predictions` saves a numpy array of sampled predictions to a pickle file
    in a specified folder.

    :param folder: The `folder` parameter is a string that represents the directory where you want to
    save the file
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file you
    want to save the sampled predictions to
    :type file_name: str
    :param sampled_predictions: The parameter "sampled_predictions" is an ndarray (NumPy array) that
    contains the sampled predictions
    :type sampled_predictions: np.ndarray
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump(sampled_predictions, f)


def load_sampled_predictions(folder: str, file_name: str) -> np.ndarray:
    """
    The function `load_sampled_predictions` loads a numpy array of sampled predictions from a pickle
    file.

    :param folder: The `folder` parameter is a string that represents the path to the folder where the
    file is located
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file you
    want to load
    :type file_name: str
    :return: a NumPy array.
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        sampled_predictions = pickle.load(f)
    return sampled_predictions


def save_gaussian_predictions(folder: str, file_name: str, mean: typing.Union[np.ndarray, float], var: typing.Union[np.ndarray, float]):
    """
    The function `save_gaussian_predictions` saves the mean and variance of Gaussian predictions to a
    pickle file in the specified folder.

    :param folder: The `folder` parameter is a string that represents the directory where you want to
    save the file
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file you
    want to save
    :type file_name: str
    :param mean: The `mean` parameter can be either a numpy array or a float. It represents the mean
    values of the Gaussian predictions
    :type mean: typing.Union[np.ndarray, float]
    :param var: The `var` parameter represents the variance of the Gaussian distribution. It can be
    either a single float value or a numpy array containing the variances for each data point
    :type var: typing.Union[np.ndarray, float]
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'wb') as f:
        pickle.dump((mean, var), f)


def load_gaussian_predictions(folder: str, file_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    The function `load_gaussian_predictions` loads mean and variance predictions from a pickle file.

    :param folder: The `folder` parameter is a string that represents the path to the folder where the
    file is located
    :type folder: str
    :param file_name: The `file_name` parameter is a string that represents the name of the file
    containing the Gaussian predictions
    :type file_name: str
    :return: a tuple containing two numpy arrays: mean and var.
    """
    folder_path = os.path.join(folder, file_name)
    with open(f'{folder_path}.pkl', 'rb') as f:
        mean, var = pickle.load(f)
    return mean, var
