import pre_process_2

import pytest

def test_get_final_loads_correctly():

    img = pre_process_2.get_final("test_image.tif", False, downsample=False)
    assert img.shape == (608, 608, 608)

def test_get_final_downsamples_correctly():

    img = pre_process_2.get_final("test_image.tif", False, downsample=True)
    assert img.shape == (128, 128, 128)