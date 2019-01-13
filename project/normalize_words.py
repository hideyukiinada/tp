#!/usr/bin/env python
"""
Normalize word list.

References
----------
process_texts() function defined in
https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging

import nltk
nltk.download('stopwords') # Download upon loading.  This method does not download if it's already cached.

from nltk.corpus import stopwords
import gensim


log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

def normalize_words(word_list):
    stop_words = stopwords.words('english')
    bigram = gensim.models.Phrases(word_list)

    # Remove stop words
    new_list = list()
    for w in word_list:
        if w not in stop_words:
            new_list.append(w)

    # Bigram
    word_list = bigram[new_list]
    new_list = list()
    for w in word_list:
        new_list.append(w)

    return new_list
