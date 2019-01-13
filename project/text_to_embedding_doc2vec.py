#!/usr/bin/env python
"""
Convert text data to embeddings using doc2vec

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
from gensim.models.doc2vec import Doc2Vec

from project.text_to_id import map_text_to_word_list

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

RESERVED_WORD_LIST = ["<UNK>", "<EOS>", "<PAD>"]
MODEL_FILE = "/tmp/tp/doc2vec_newsgroup.model"

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
    """
    model = Doc2Vec.load(MODEL_FILE)

    x_list = list()
    y_list = list()

    for i, text in enumerate(text_list):
        log.debug("Processing post: [%d]" % (i + 1))
        word_list = text.split()

        v = model.infer_vector(word_list)

        # For now, do not change non-zero element to 1.
        label_id = label_to_id[label_for_text_list[i]]
        label_id = keras.utils.to_categorical(label_id, num_labels).astype(np.float32)
        label_id = label_id.reshape(1, num_labels)

        x_list.append(v)
        y_list.append(label_id)

    x = np.array(x_list)
    print(x.shape)
    y = np.concatenate(y_list)

    return x, y