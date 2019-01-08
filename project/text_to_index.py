#!/usr/bin/env python
"""
Convert text data to indices

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
import re

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

def label_to_index(labels):
    """
    Generate an index for each unique label in the list. Returns a dictionary to map between them.

    Parameters
    ----------
    labels: list of str
        List of labels

    Returns
    -------
    label_to_index: dict
        Unique label to integer index mapping
    index_to_label: dict
        Integer index to unique label mapping
    """

    unique_labels = sorted(set(labels))

    label_to_index = {l: i for i, l in enumerate(unique_labels)}
    index_to_label = {i: l for l, i in label_to_index.items()}

    return label_to_index, index_to_label

def text_to_word_list(text, filters='!"#$%&()*+,-./:;<=>?@[]^_`{|}~\'', separator_list=None):
    """
    Convert text to word list

    Parameters
    ----------
    text: str
        Text

    filters: string
        A string specifying characters to remove.  Each character will be removed from output.

    separators: list of strings
        A list of strings containing one or more separators
        if not specified, ["\n", " "] will be used.

    Returns
    -------
    word: list of str
        List of words
    """

    re_special_characters="-.*+?^$|[]()\\" # This is a subset that requires escaping with \ for using in re.sub()

    filter_char_list = list(filters) # list of chars

    escaped_filter_char_list = list()
    for c in filter_char_list:
        if c in re_special_characters:
            c = "\\" + c
        e = c

        escaped_filter_char_list.append(e)

    adjusted_filter = "[" + "|".join(escaped_filter_char_list) + "]"
    if separator_list is None:
        separator_list = ["\n", " "]

    first_separator = separator_list[0]

    for i, s in enumerate(separator_list):
        if i == 0: # skip replacing the first separator
            continue
        text = text.replace(s, first_separator)

    # Replace multiple separators with 1 separator
    exp = first_separator + "+"
    reg = re.compile(exp)
    text = reg.sub(first_separator, text)

    # Remove characters
    reg = re.compile(adjusted_filter)
    text = reg.sub("", text)

    word_list = text.split(first_separator)

    return word_list
