#!/usr/bin/env python
"""
Unit test use case of a method that is used in tp.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import unittest
import os
import logging
from pathlib import Path

import numpy as np

from project.text_to_index import label_to_index
from project.text_to_index import text_to_word_list

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


class TestTextToIndex(unittest.TestCase):

    def test_label_to_index(self):
        """
        Test label to index mapping
        """

        labels = ["banana", "apple", "coconut", "apple"]
        label2index, index2label = label_to_index(labels)

        actual = sorted(label2index.keys())
        expected = ["apple", "banana", "coconut"]

        actual = sorted(index2label.keys())
        expected = [0, 1, 2]

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

    def test_text_to_word_list(self):
        """
        Test text to word list mapping
        """
        text = "apple       banana\n coconut"
        word_list = text_to_word_list(text)

        actual = sorted(word_list)
        expected = ["apple", "banana", "coconut"]

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

    def test_text_to_word_list2(self):
        """
        Test text to word list mapping
        """

        filter = '!"#$%&()*+,-./:;<=>?@[]^_`{|}~\''
        other_symbols = "[]"

        char_to_check = filter + other_symbols

        for c in char_to_check:
            s = "a" + c + "b"

            expected = 3
            actual = len(s)

            result = actual == expected
            self.assertTrue(result,
                            "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

            text = s
            s2 = text_to_word_list(text)
            expected = "ab"
            actual = s2[0]

            result = actual == expected
            self.assertTrue(result,
                            "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))


def main():
    """Invoke test function"""

    unittest.main()


if __name__ == "__main__":
    main()
