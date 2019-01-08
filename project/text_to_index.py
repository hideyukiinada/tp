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

def text_list_to_word_list(text_list, filters='!"#$%&()*+,-./:;<=>?@[]^_`{|}~\'', separator_list=None):
    """
    Convert text to word list

    Parameters
    ----------
    text: list of str
        List of text

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

    word_list = list()

    for text in text_list:
        w = text_to_word_list(text, filters=filters, separator_list=separator_list)
        word_list += w

    return word_list

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

def word_list_to_vocabulary(word_list, top_vocabulary_size):
    """
    From the list of words, select the top vocabulary with the size set to vocabulary_size, and returns
    word to index map as well as index to word map for the vocabulary.

    Parameters
    ----------
    word_list: list of str
        List of words
    top_vocabulary_size: int
        Size of top vocabulary

    Returns
    -------
    vocabulary: list
        vocabulary list
    vocabulary_size: int
        Size of the vocabulary
    top_vocabulary: list
        Top vocabulary list
    top_vocabulary_size: int
        Size of the top vocabulary
    word_to_index: dict
        Word to index to top vocabulary mapping
    index_to_word: dict
        Index to top vocabulary to word mapping
    """
    # Count occurrence
    word_count = dict()
    for w in word_list:
        if w not in word_count:
            word_count[w] = 0

        count = word_count[w]
        word_count[w] = count + 1

    # The unique words in across all the files
    vocabulary = sorted(word_count.keys(), key=lambda v: word_count[v], reverse=True)
    vocabulary_size = len(vocabulary)
    log.info("Size of vocabulary: %d" % (vocabulary_size))
    top_vocabulary_size = min(vocabulary_size, top_vocabulary_size)
    log.info("Size of top vocabulary: %d" % (top_vocabulary_size))

    top_vocabulary = vocabulary[:top_vocabulary_size]

    # print top 10
    log.info("Top 5 words (count)")
    for i in range(5):
        log.info("%10s (%d)" % (top_vocabulary[i], word_count[top_vocabulary[i]]))

    # Mapping between words and IDs
    word_to_index = {w: i + 1 for i, w in enumerate(top_vocabulary)}  # +1 for unknown
    index_to_word = {i: w for w, i in word_to_index.items()}

    keys = word_to_index.keys()
    for i, k in enumerate(keys):
        log.info("%d for %s" % (word_to_index[k], k))
        if i + 1 == 5:
            break

    keys = index_to_word.keys()
    for i, k in enumerate(keys):
        log.info("%s for %d" % (index_to_word[k], k))
        if i + 1 == 5:
            break

    return vocabulary, vocabulary_size, top_vocabulary, top_vocabulary_size, word_to_index, index_to_word