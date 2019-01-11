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
from project.load_text_data import load_text_from_files
from project.text_to_id import map_label_to_id

TEST_DATA_DIR = Path("/tmp/tp/test")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def create_test_files():
    """
    Create test files under the test data directory.
    """

    test_sample_dir = TEST_DATA_DIR

    if test_sample_dir.exists() is False:
        test_sample_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    dirs = ["apple",
            "banana",
            "cherry"]
    for dir in dirs:
        dir = test_sample_dir / Path(dir)
        if dir.exists() is False:
            dir.mkdir(parents=True, exist_ok=True)
            log.info("%s created" % (dir))
        else:
            log.info("%s found. Skip creation" % (dir))

    # Create files
    files = ["apple/granny_smith_apple.txt",
             "apple/fuji_apple.txt",
             "apple/golden_delicious.txt",
             "apple/gala.txt",
             "apple/honeycrisp.txt",
             "banana/banana.txt",  # Missing test data
             "cherry/bing_cherry.txt",  # 1 test data
             "cherry/rainier_cherry.txt"]
    text_body = ["Granny Smith apples are green.",
                 "Fuji apples are red.",
                 "I love Golden Delicious apples.",
                 "Gala apples are my favorite.",
                 "I want to try an apple pie made out of Honeycrisp apples",
                 "Bananas are good for breakfast.",
                 "Bing cherries are popular.",
                 "Rainer cherries are in season right now."]

    for i, f in enumerate(files):
        text_path = test_sample_dir / Path(f)
        if text_path.exists() is False:
            with open(text_path, "w") as fh:
                fh.write(text_body[i])
            log.info("%s written" % (text_path))


class TestLoadTextData(unittest.TestCase):

    def test_load_text(self):
        """
        Test loading text
        """
        create_test_files()

        (x_train, y_train), (x_test, y_test) = load_text_from_files(TEST_DATA_DIR, test_dataset_ratio=0.2)

        # x_train len
        expected = 5  # 4 apples and 1 cherry
        actual = len(x_train)

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

        # x_test len
        expected = 2  # 1 apples and 1 cherry
        actual = len(x_test)

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

    def test_load_text_no_test(self):
        """
        Test loading text
        """
        create_test_files()

        (x_train, y_train), (x_test, y_test) = load_text_from_files(TEST_DATA_DIR, test_dataset_ratio=0)

        # x_train len
        expected = 8  # 5 apples, 1 banana and 2 cherries
        actual = len(x_train)

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

        # x_train len
        expected = 0
        actual = len(x_test)

        result = actual == expected
        self.assertTrue(result, "actual does not match expected. \nActual:\n%s, \nExpected:\n%s" % (actual, expected))

        # Check to see if text is loaded.
        result = False
        for x in x_train:
            if x == "Granny Smith apples are green.":
                result = True
                break

        self.assertTrue(result, "Text was not loaded.")


def main():
    """Invoke test function"""

    unittest.main()


if __name__ == "__main__":
    main()
