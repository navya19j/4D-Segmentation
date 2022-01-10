#!/usr/bin/env python3

import ast
import glob
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw
import time
from tifffile import *
from tifffile.tifffile import imread

st.set_page_config(
    page_title="4D Endosome Segmentation", page_icon=":microscope:", layout="wide"
)
st.title("4D Segmentation Visualisation")


@st.cache
def load_data(fname):
    file = open(fname, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)

    return dictionary


@st.cache
def extract_as_dataframe(MAX_ITER=500):
    df = pd.DataFrame()

    for i, track_id in enumerate(dictionary.keys()):
        for v in dictionary[track_id]:
            t = v[0]
            x, y, z, w, h, d = v[1]
            # st.write(f"time: {t}")
            # st.write(f"data: {x}, {y}, {z}, {w}, {h}, {d}")

            data_dict = {
                "track_id": str(track_id),
                "t": t,
                "x": x,
                "y": y,
                "z": z,
                "w": w,
                "h": h,
                "d": d,
            }

            df = df.append(pd.DataFrame.from_records([data_dict]))

            # st.write(data_dict)

        if i == MAX_ITER:
            break
    return df


################# RUN ##############

#### LOAD DATA

st.sidebar.header("Data")
filenames = glob.glob("./output/*/")
CELL_NAME = st.sidebar.selectbox("Select a cell: ", filenames)
fname = os.path.join(CELL_NAME, "predicted_mask", "object.txt")
fname = st.sidebar.text_input("Select an object file: ", fname)

# NOTE: can load all the items, but the plotting struggles with so many. we need to filter it before hand

dictionary = load_data(fname)

# MAX_ITEMS_TO_LOAD = st.number_input("Maximum items to load")
df = extract_as_dataframe(MAX_ITER=2000)#len(dictionary))

st.sidebar.write(f"Loaded {len(dictionary)} tracked objects ({len(df)} data points).")

st.header("Raw Object Tracking Data")
st.write(df)
st.subheader("Object Tracking Statistics")
st.write(df.describe())

##############################################################################################################################
#### FILTER DATA
st.sidebar.subheader("Filter Options")
# minimum number of tracks
# TODO: change to based on actual data
df_track_minimum_length = (
    df.groupby("track_id").count().sort_values(by="t", ascending=False)
)
max_tracks = df_track_minimum_length["t"].max()
mean_tracks = int(df_track_minimum_length["t"].mean())
MIN_TRACKS = st.sidebar.select_slider(
    "Minimum Track Length", range(0, max_tracks), value=int(0.75 * max_tracks)
)
min_track_ids = list(
    df_track_minimum_length[df_track_minimum_length["t"] > MIN_TRACKS].index
)

df_filter = df[df["track_id"].isin(min_track_ids)]
# df_filter["size"] = df_filter["w"] * df_filter["h"] * df_filter["d"]

uniq_ids = df_filter["track_id"].unique()

st.sidebar.subheader("Statistics")
st.sidebar.write(
    f"Filtered to {len(uniq_ids)} individual tracks ({len(df_filter)} data points)."
)


############################################################################################################
##### 2D IMAGE PLOTS

# TODO: not working properly with large amount of tracks
# st.markdown("""---""")
# st.header("Images")

# from viz_utils import *

# img = load_img_as_np_array(os.path.join(CELL_NAME, "predicted_mask", "1.tif"))

# st.write("SHAPE", img.shape)
# T_MAX = 198
# # Z_MAX = img.

# # IMG_SHAPE = (600, 600, img.shape)
# base_img = np.zeros(shape=img.shape)

# idx = st.slider("Select slice", 0, base_img.shape[2], 0)


# st.subheader("Slice Data")

# img_bbox = draw_bounding_boxes(df_filter, base_img, idx)

# animate_button = st.button("Animate Image")
# progress_bar = st.empty()

# animation_header = st.markdown(f"**Slice {idx}/{base_img.shape[2]}**")
# img_cols  = st.columns(2)
# df_placeholder = img_cols[0].dataframe(df_filter[df_filter["z"]==idx])
# bbox_placeholder = img_cols[1].image(img_bbox, clamp=True, caption="Tracked Objects", width=500)

# if animate_button:
#     progress_bar.progress(0)
#     for idx in range(base_img.shape[2]):
#         df_img = df_filter[df_filter["z"] == idx]
#         img_bbox = draw_bounding_boxes(df_filter, base_img, idx)

#         animation_header.markdown(f"Slice: {idx}/{base_img.shape[2]}")
#         df_placeholder.dataframe(df_filter[df_filter["z"]==idx])
#         bbox_placeholder.image(img_bbox, clamp=True, caption=f"Slice : {idx}", width=500)
#         time.sleep(0.5)

#         progress_bar.progress((idx + 1) / base_img.shape[2])


#### PLOTTING
st.markdown("---")
st.header("Plots")
# need to flip some axes due to being in img coordinates
fig_xyz = px.scatter_3d(
    df_filter, x="z", y="x", z="y", color="track_id", title="XYZ Coordinates" #,size=size
)


fig_xyt = px.scatter_3d(
    df_filter, x="t", y="x", z="y", color="track_id",  title="XYT Coordinates" #, size=size
) 

plot_cols = st.columns(2)
plot_cols[0].plotly_chart(fig_xyz)
plot_cols[1].plotly_chart(fig_xyt)

############################################################################################################
# Distance Travelled

st.markdown("---")
st.header("Distance Travelled")

# filter_id = st.selectbox("Select an ID: ", df_filter["track_id"].unique())

df_distance = pd.DataFrame()

for filter_id in df_filter["track_id"].unique():

    # filter for specific id
    df_filter_distance = df_filter[df_filter["track_id"]==filter_id].sort_values(by="t", ascending=True)

    # difference in cartesian coordinates
    df_filter_distance["x_diff"] = df_filter_distance['x'].diff()
    df_filter_distance["y_diff"] = df_filter_distance['y'].diff()
    df_filter_distance["z_diff"] = df_filter_distance['z'].diff()
    df_filter_distance = df_filter_distance.fillna(0)

    # euclidean distance
    df_filter_distance["dist"] = np.sqrt(
        np.power(df_filter_distance["x_diff"], 2) +
        np.power(df_filter_distance["y_diff"], 2) + 
        np.power(df_filter_distance["z_diff"], 2)
    )

    # cumulative distance
    df_filter_distance["cumulative_dist"] = df_filter_distance["dist"].cumsum(axis=0)   
    
    # append to full dataframe
    df_distance = df_distance.append(df_filter_distance)


st.dataframe(df_distance)

# plot distance data
fig_dist = px.line(df_distance, x="t", y="dist", color="track_id", title="Euclidean Distance")
fig_cumulative_dist = px.line(df_distance, x="t", y="cumulative_dist", color="track_id", title="Cumulative Euclidean Distance")

dist_cols = st.columns(2)
dist_cols[0].plotly_chart(fig_dist, use_container_width=True)
dist_cols[1].plotly_chart(fig_cumulative_dist, use_container_width=True)

# TODO: dual axis plots for dist and cumdist
# https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express

# need to do it per id
# https://stackoverflow.com/questions/13114512/calculating-difference-between-two-rows-in-python-pandas
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
# https://en.wikipedia.org/wiki/Euclidean_distance
# easier to do it manually with the diff between rows..

############################################################################################################
##### Volumetric Data

st.markdown("---")
st.header("Volumetric Data")

# calculate volumetric data
df_distance["size"] = df_distance["w"] * df_distance["h"] * df_distance["d"]
df_distance['avg_size'] = df_distance.groupby("track_id")["size"].transform('mean')
df_distance["dist_per_avg_size"] = df_distance["dist"] / df_distance["avg_size"]
df_distance["dist_per_size"] = df_distance["dist"] / df_distance["size"]
df_distance["cumulative_dist_per_size"] = df_distance["cumulative_dist"] / df_distance["size"]
df_distance["cumulative_dist_per_avg_size"] = df_distance["cumulative_dist"] / df_distance["avg_size"]
st.write(df_distance)

# volumetric distribution
fig_hist = px.histogram(df_distance, x="size", color="track_id", nbins=50, title="Size Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

# plot volumetric distance data
fig_dist_size = px.line(df_distance, x="t", y="dist_per_size", color="track_id", title="Distance per Size")
fig_dist_avg_size = px.line(df_distance, x="t", y="dist_per_avg_size", color="track_id", title="Distance per Average Size")
fig_cumulative_dist_per_size = px.line(df_distance, x="t", y="cumulative_dist_per_size", color="track_id", title="Cumulative Distance per Size")
fig_cumulative_dist_per_avg_size = px.line(df_distance, x="t", y="cumulative_dist_per_avg_size", color="track_id", title="Cumulative Distance per Avg Size")

dist_cols = st.columns(2)
dist_cols[0].plotly_chart(fig_dist_size, use_container_width=True)
dist_cols[1].plotly_chart(fig_cumulative_dist_per_size, use_container_width=True)
dist_cols[0].plotly_chart(fig_dist_avg_size, use_container_width=True)
dist_cols[1].plotly_chart(fig_cumulative_dist_per_avg_size, use_container_width=True)



# save to csv
df_distance.to_csv("data.csv")

############################################################################################################