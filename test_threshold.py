#!usr/bin/python3

import pytest

from threshold import apply_threshold

import numpy as np

def test_threshold_is_correct_shape():

    img = np.ones((3, 3, 1))

    threshold_img = apply_threshold(img)

    assert img.shape == threshold_img.shape
    assert threshold_img.dtype == np.uint8

    
def test_threshold_is_correct():

    img = np.ones((3, 3, 1))
    img[0, 0, 0] = 0

    threshold_img = apply_threshold(img)

    assert np.array_equal(img, threshold_img)
