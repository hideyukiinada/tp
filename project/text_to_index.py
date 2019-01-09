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

RESERVED_WORD_LIST = ["<UNK>", "<EOS>", "<PAD>"]

def map_label_to_index(labels):
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

def map_text_list_to_word_list(text_list, filters='!"#$%&()*+,-./:;<=>?@[]^_`{|}~\'', separator_list=None):
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
        w = map_text_to_word_list(text, filters=filters, separator_list=separator_list)
        word_list += w

    return word_list

def map_text_to_word_list(text, filters='!"#$%&()*+,-./:;<=>?@[]^_`{|}~\'', separator_list=None):
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

def map_word_list_to_vocabulary(word_list, top_vocabulary_size):
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
    reserved_word_size: int
        Size of reserved word list
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

    top_vocabulary = vocabulary[:top_vocabulary_size] + RESERVED_WORD_LIST
    for i in range(len(RESERVED_WORD_LIST)):
        word_count[top_vocabulary[-i]] = -1 # Set RESERVED_WORD_LIST entries count to -1

    # print top ranking words
    log.info("Top 5 words (count)")
    for i in range(5):
        log.info("%10s (%d)" % (top_vocabulary[i], word_count[top_vocabulary[i]]))

    # Mapping between words and IDs
    word_to_index = {w: i for i, w in enumerate(top_vocabulary)}
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

    return vocabulary, vocabulary_size, top_vocabulary, top_vocabulary_size, len(RESERVED_WORD_LIST), \
           word_to_index, index_to_word


def map_text_to_token_matrix(text_list, label_for_text_list, top_vocabulary_size, reserved_word_size, num_labels, label_to_index, word_to_index):
    """

    Parameters
    ----------
    text_list: list of str
        List of text
    label_for_text_list: list of str
        List of labels, which is the ground truth for each text on the text_list
    top_vocabulary_size: int
        Size of the top vocabulary
    reserved_word_size: int
        Size of reserved word list
    num_labels:
        Number of labels
    label_to_index: dict
        Label to integer index mapping
    word_to_index: dict
        Word to index to top vocabulary mapping

    Returns
    -------
    x: ndarray
        Numpy array of indices representing text
    y: ndarray
        Numpy array of indices representing labels
    """
    x_list = list()
    y_list = list()

    top_and_reserved_vocabulary_size = top_vocabulary_size + reserved_word_size

    for i, text in enumerate(text_list):
        log.debug("Processing post: [%d]" % (i + 1))
        words_in_text = map_text_to_word_list(text)

        word_id_list = list()
        for w in words_in_text:
            if w not in word_to_index:
                id = 0  # Unknown
            else:
                id = word_to_index[w]
            word_id_list.append(id)

        word_array = np.array(word_id_list)
        word_array_one_hot = keras.utils.to_categorical(word_array, top_and_reserved_vocabulary_size).astype(np.float32)

        s = np.sum(word_array_one_hot, axis=0)
        s = s.reshape(1, top_and_reserved_vocabulary_size)

        x_list.append(s)

        # For now, do not change non-zero element to 1.
        label_index = label_to_index[label_for_text_list[i]]
        label_index = keras.utils.to_categorical(label_index, num_labels).astype(np.float32)
        label_index = label_index.reshape(1, num_labels)
        y_list.append(label_index)

    x = np.concatenate(x_list, axis=0)
    print(x.shape)
    y = np.concatenate(y_list)

    return x, y

def map_text_to_word_index(text_list, label_for_text_list, top_vocabulary_size, reserved_word_size, num_labels, label_to_index, word_to_index):
    """

    Parameters
    ----------
    text_list: list of str
        List of text
    label_for_text_list: list of str
        List of labels, which is the ground truth for each text on the text_list
    top_vocabulary_size: int
        Size of the top vocabulary
    reserved_word_size: int
        Size of reserved word list
    num_labels:
        Number of labels
    label_to_index: dict
        Label to integer index mapping
    word_to_index: dict
        Word to index to top vocabulary mapping

    Returns
    -------
    x: ndarray
        Numpy array of word indices. Each entry is an index to the vocabulary list.
    y: ndarray
        Numpy array of indices representing labels
    """
    x_list = list()
    y_list = list()

    top_and_reserved_vocabulary_size = top_vocabulary_size + reserved_word_size

    for i, text in enumerate(text_list):
        log.debug("Processing post: [%d]" % (i + 1))
        words_in_text = map_text_to_word_list(text)

        word_id_list = list()
        for w in words_in_text:
            if w not in word_to_index:
                id = 0  # Unknown
            else:
                id = word_to_index[w]
            word_id_list.append(id)

        # For now, do not change non-zero element to 1.
        label_index = label_to_index[label_for_text_list[i]]
        label_index = keras.utils.to_categorical(label_index, num_labels).astype(np.float32)
        label_index = label_index.reshape(1, num_labels)
        x_list.append(word_id_list)
        y_list.append(label_index)

    x = np.array(x_list)
    print(x.shape)
    y = np.concatenate(y_list)

    return x, y