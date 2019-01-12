#!/usr/bin/env python
"""
Convert text data to embeddings

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging
import re

import numpy as np
import keras
from gensim.models.word2vec import Word2Vec

from project.text_to_id import map_text_to_word_list

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

RESERVED_WORD_LIST = ["<UNK>", "<EOS>", "<PAD>"]
MODEL_PATH = "/tmp/tp/word2vec_example.model"

def map_text_list_to_embedding(text_list, label_for_text_list, num_labels, label_to_id):
    """

    Parameters
    ----------
    text_list: list of str
        List of text
    label_for_text_list: list of str
        List of labels, which is the ground truth for each text on the text_list
    num_labels:
        Number of labels
    label_to_id: dict
        Label to integer id mapping

    Returns
    -------
    x: ndarray
        Numpy array of mean word embeddings for each text.
    y: ndarray
        Numpy array of indices representing labels
    missing_words: set
        Set of words not in the Word2Vec model's dictionary.
    """
    model = Word2Vec.load(MODEL_PATH)
    missing_words = set()
    x_list = list()
    y_list = list()

    total_found_in_dict = 0
    total_not_in_dict = 0
    for i, text in enumerate(text_list):
        log.debug("Processing post: [%d]" % (i + 1))
        words_in_text = map_text_to_word_list(text)

        word_v_list = list()
        for w in text:
            try:
                v = model[w]
            except KeyError:
                missing_words.add(w)
                #log.warning("Skipping %s" % (w))
                total_not_in_dict += 1
                continue

            word_v_list.append(v)
            total_found_in_dict += 1

        if len(word_v_list) == 0:
            # log.warning("Did not find any words in vocabulary.  Skipping the text.")
            continue

        # For now, do not change non-zero element to 1.
        label_id = label_to_id[label_for_text_list[i]]
        label_id = keras.utils.to_categorical(label_id, num_labels).astype(np.float32)
        label_id = label_id.reshape(1, num_labels)

        # Squish word_id_list
        word_v_np = np.array(word_v_list)
        word_count = word_v_np.shape[0]
        word_v_mean = np.sum(word_v_np, axis=0)/word_count
        #log.info("word_v_mean.shape")
        #log.info(word_v_mean.shape)

        x_list.append(word_v_mean)
        y_list.append(label_id)

    x = np.array(x_list)
    print(x.shape)
    y = np.concatenate(y_list)

    log.info("Number of words found in dict: %d" % (total_found_in_dict))
    log.info("Number of words not found in dict: %d" % (total_not_in_dict))

    return x, y, missing_words