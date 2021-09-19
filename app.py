#!/usr/bin/env python3

import ast
import glob

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
filenames = glob.glob("./**/object.txt")
fname = st.sidebar.text_input("Select an object file: ", filenames[0])

# NOTE: can load all the items, but the plotting struggles with so many. we need to filter it before hand

dictionary = load_data(fname)

# MAX_ITEMS_TO_LOAD = st.number_input("Maximum items to load")
df = extract_as_dataframe(MAX_ITER=len(dictionary))

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
##### 2D IMAGE PLOTS

# TODO: 2D plotting visualisation for bounding boxes
# loop through the object
# plot each box as rectangel in image, color is track id

# TODO: i need to understand better how the image dimensions and time work? bit confused about if the image is xyz or xyt?
# which dimeansions are what?

st.markdown("""---""")
st.header("Images")


def draw_bounding_boxes(df, img_bbox, idx):
    
    df_img = df[df["z"] == idx]

    im = Image.fromarray(img_bbox[:, :, idx])
    im = im.convert("RGB")

    draw = ImageDraw.Draw(im)

    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    disp_ids = []
    for i in range(len(df_img)):
        row = df_img.iloc[i]
        id, t, x, y, z, w, h, d = row

        col = colors[int(id) % len(colors)]
        draw.rectangle((x, y, x+w, y+h), fill=col)

        if id not in disp_ids:
            draw.text((x, y-20), text=f"id: {id}", fill="white") 
            disp_ids.append(id)

    # TODO: double check how the boxes are defined? assume, x0, y0

    img_bbox = np.array(im)
    return img_bbox


T_MAX = 198
Z_MAX = 66

IMG_SHAPE = (600, 600, Z_MAX)
base_img = np.zeros(shape=IMG_SHAPE)

idx = st.slider("Select slice", 0, base_img.shape[2], 0)

colors = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "gray"]
# df_img["col"] = df["track_id"].astype(int).apply(colors[df["track_id"] % len(colors)]) # color
st.subheader("Slice Data")

img_bbox = draw_bounding_boxes(df_filter, base_img, idx)

animate_button = st.button("Animate Image")
progress_bar = st.empty()

animation_header = st.markdown(f"**Slice {idx}/{base_img.shape[2]}**")
img_cols  = st.columns(2)
df_placeholder = img_cols[0].dataframe(df_filter[df_filter["z"]==idx])
bbox_placeholder = img_cols[1].image(img_bbox, clamp=True, caption="Tracked Objects", width=500)

if animate_button:
    progress_bar.progress(0)
    for idx in range(base_img.shape[2]):
        df_img = df_filter[df_filter["z"] == idx]
        img_bbox = draw_bounding_boxes(df_filter, base_img, idx)

        animation_header.markdown(f"Slice: {idx}/{base_img.shape[2]}")
        df_placeholder.dataframe(df_filter[df_filter["z"]==idx])
        bbox_placeholder.image(img_bbox, clamp=True, caption=f"Slice : {idx}", width=500)
        time.sleep(0.5)

        progress_bar.progress((idx + 1) / base_img.shape[2])



############################################################################################################

######### VISUALISING RAW IMAGE, LABELS and PREDICTIONS
st.markdown("""---""")
st.header("Predicted Images")
# TODO:
# Set fixed limits for the axes
# load images for raw, label, predicted
# match up with image data

@st.cache
def load_img_as_np_array(path):
    """Load a multidimensional tiff image as np array"""
    img_array = []
    
    img = imread(path)
    d = img.shape[0]

    for i in range (0,d):
        m_new = imread(path,key=i)
        img_array.append(np.array(m_new, dtype=np.float32))

    k = np.amax(img_array)
    img_array = np.array(img_array)
    img_array = (img_array)/float(k)
    return img_array



img_path = st.text_input("Select an image", "Data_Resized/cell02_EEA1 TagRFP_T005.tif")
mask_path = st.text_input("Select a label mask", "Labeled_Resized/cell02_EEA1 TagRFP_T005.tif")

img_array = load_img_as_np_array(img_path)
mask_array = load_img_as_np_array(mask_path)

idx = st.slider("Select slice", 0, img_array.shape[2], 0)
img_animation_button = st.button("Animate Images")
img_header = st.markdown(f"Slice : {idx}")

cols = st.columns(3)
col_0 = cols[0].image(img_array[:, :, idx], caption="raw image", use_column_width=True)
col_1 = cols[1].image(mask_array[:, :, idx], caption="label image", use_column_width=True)
col_2 = cols[2].image(mask_array[:, :, idx], caption="predicted image", use_column_width=True)



if img_animation_button:

    for idx in range(img_array.shape[2]):
        img_header.markdown(f"Slice : {idx}/{img_array.shape[2]}")
        col_0.image(img_array[:, :, idx], caption="raw image", use_column_width=True)
        col_1.image(mask_array[:, :, idx], caption="label image", use_column_width=True)
        col_2.image(mask_array[:, :, idx], caption="predicted image", use_column_width=True)
        time.sleep(0.2)

st.write("NOTE: No predicted mask avaialble atm. Need to be added")

# TODO: get actuall predictions


# Theme:
# [theme]
# base="dark"
# primaryColor="#33f6ec"
