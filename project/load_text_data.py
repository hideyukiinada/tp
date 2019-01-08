#!/usr/bin/env python
"""
Load text data

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging

import tensorflow as tf
import numpy as np
import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import sklearn.datasets

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def load_text_from_files(root_dir, test_dataset_ratio=0.2, errors=None):
    """
    Load text from files under label directories.

    Parameters
    ----------
    root_dir: pathlib.Path
        Parent directory of labeled directories, each of which contains text files
    test_dataset_ratio: float
        Ratio of test dataset to split the data into training dataset and test dataset
    errors: str
        Set to 'ignore' if you want to ignore a byte that cannot be decoded.

    Returns
    -------
    (x_train, y_train), (x_test, y_test): tuples of list
        x_train and x_test contains the list of text in string.
        y_train and y_test contains the list of labels in string.

    Raises
    ------
    ValueError
        If root_dir does not exist.
    """
    root_dir_path = root_dir
    if root_dir_path.exists() is False:
        log.fatal("% does not exist." % (root_dir_path))
        raise ValueError("% does not exist." % (root_dir_path))

    label_dirs = [x for x in root_dir_path.iterdir() if x.is_dir()]

    all_training_samples = list()
    all_test_samples = list()

    for label_dir in label_dirs:

        label = label_dir.name
        log.info("Scanning %s" % (label))

        samples_for_label = list()
        # Process files
        for input_file in label_dir.glob("*"):
            with open(input_file, "r", errors=errors) as f:
                text = f.read()  # Change this if you are reading a very large text file.
                sample = (text, label)
                samples_for_label.append(sample)

        num_samples_for_label = len(samples_for_label)
        if len == num_samples_for_label:
            continue

        if num_samples_for_label == 1 and test_dataset_ratio > 0:
            log.info("Label %s contain only 1 sample and cannot be split to training and test dataset. Skipping.")
            continue

        if test_dataset_ratio > 0:
            num_test_samples = max(int(num_samples_for_label * test_dataset_ratio), 1)
        else:
            num_test_samples = 0

        num_training_samples = num_samples_for_label - num_test_samples
        training_samples = samples_for_label[:num_training_samples]
        test_samples = samples_for_label[num_training_samples:]

        all_training_samples += training_samples
        all_test_samples += test_samples

    # Unzip the list
    (x_train, y_train) = tuple(zip(*all_training_samples))

    if len(all_test_samples) > 0:
        (x_test, y_test) = tuple(zip(*all_test_samples))
    else:
        x_test = list()
        y_test = list()

    return (x_train, y_train), (x_test, y_test)
