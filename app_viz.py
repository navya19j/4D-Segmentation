#!/usr/bin/env python3

import streamlit as st
import numpy as np
from tifffile.tifffile import imread
import time
import glob

from viz_utils import *

######### VISUALISING RAW IMAGE, LABELS and PREDICTIONS

st.set_page_config(layout="wide")
st.title("4D Segmentation Visualisation")
# TODO:
# Set fixed limits for the axes
# load images for raw, label, predicted
# match up with image data



img_folder = st.text_input("Select the raw image directory", "Data/cell02_APPL1_GFP/")
mask_folder = st.text_input("Select the labelled mask directory", "Data/cell02_APPL1_GFP/") # TODO: test on real masks
pred_folder = st.text_input("Select the predicted mask directory", "predicted_mask_test/")


imgs = sorted(glob.glob(img_folder + "*.tif"))
masks = sorted(glob.glob(mask_folder + "*.tif"))
preds = sorted(glob.glob(pred_folder + "*.tif"))

MAX_TIMESTEP = min(len(imgs), len(masks), len(preds))
imgs = imgs[:MAX_TIMESTEP]
masks = masks[:MAX_TIMESTEP]
preds = preds[:MAX_TIMESTEP]

st.write(len(imgs), len(masks), len(preds))

t_idx = st.slider("Select a timestep", 0, MAX_TIMESTEP)



img, mask, pred = load_images(imgs, masks, preds, t_idx)

MAX_Z_HEIGHT = min(img.shape[2], mask.shape[2], pred.shape[2])
z_idx = st.slider("Select a z-height", 0, MAX_Z_HEIGHT)

button_cols = st.columns(2)
time_animation_button = button_cols[0].button("Animate Image Over Time")
time_header = button_cols[0].markdown(f"Timestep: {t_idx}/{MAX_TIMESTEP}")
height_animation_button = button_cols[1].button("Animate Image Over Height")
height_header = button_cols[1].markdown(f"Z-Height : {z_idx}/{MAX_Z_HEIGHT}")

cols = st.columns(3)
col_0 = cols[0].image(img[:, :, z_idx], clamp=True, caption="raw image", use_column_width=True)
col_1 = cols[1].image(mask[:, :, z_idx], clamp=True, caption="label image", use_column_width=True)
col_2 = cols[2].image(pred[:, :, z_idx], clamp=True, caption="predicted image", use_column_width=True)


if time_animation_button:

    for t_idx in range(0, MAX_TIMESTEP):
        img, mask, pred = load_images(imgs, masks, preds, t_idx)
        time_header.markdown(f"Timestep: {t_idx}/{MAX_TIMESTEP}")
        col_0.image(img[:, :, z_idx], clamp=True, caption="raw image", use_column_width=True)
        col_1.image(mask[:, :, z_idx], clamp=True, caption="label image", use_column_width=True)
        col_2.image(pred[:, :, z_idx], clamp=True, caption="predicted image", use_column_width=True)
        time.sleep(0.5)

if height_animation_button:

    for z_idx in range(MAX_Z_HEIGHT):
        height_header.markdown(f"Z-Height : {z_idx}/{MAX_Z_HEIGHT}")
        col_0.image(img[:, :, z_idx], clamp=True, caption="raw image", use_column_width=True)
        col_1.image(mask[:, :, z_idx], clamp=True, caption="label image", use_column_width=True)
        col_2.image(pred[:, :, z_idx], clamp=True, caption="predicted image", use_column_width=True)
        time.sleep(0.2)


# Theme:
# [theme]
# base="dark"
# primaryColor="#33f6ec"