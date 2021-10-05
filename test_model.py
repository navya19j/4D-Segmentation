#!/usr/bin/env python3

import pytest
from model import SegmentationModel
import torch

# TODO: fixtures
def test_load_model():

    model = SegmentationModel()

    assert model.mode == "train"



