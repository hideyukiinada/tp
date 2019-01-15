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
from project.load_brown_corpus import load_text_and_label_id_from_files

TEST_DATA_DIR = Path.home() / Path("nltk_data/corpora/brown") # This requires Python 3.5+

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


class TestLoadBrownCorpus(unittest.TestCase):

    def test_load_text_and_label_id_from_files(self):
        """
        Test loading text
        """

        (x_train, y_train), (x_test, y_test), (label_to_id, id_to_label) = load_text_and_label_id_from_files(
            TEST_DATA_DIR, test_dataset_ratio=0.2)

        # x_train len
        expected = 501
        actual = len(x_train) + len(x_test)

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))


def main():
    """Invoke test function"""

    unittest.main()


if __name__ == "__main__":
    main()
