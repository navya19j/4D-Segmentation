#!/usr/bin/env python3

from numpy.core.fromnumeric import shape
import pytest
from model import SegmentationModel
import torch
import numpy as np

# TODO: fixtures
def test_load_model():

    model = SegmentationModel()

    assert model.mode == "train"


def test_pre_process():

    model = SegmentationModel()

    img = torch.zeros((180, 600))
    img_t = model.pre_process(img)

    assert img_t.shape == (180, 600) # this is wrong
    assert img_t.dtype == torch.float32

