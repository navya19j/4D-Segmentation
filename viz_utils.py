
import streamlit as st
import numpy as np
from tifffile import imread
from PIL import Image, ImageDraw


@st.cache
def load_img_as_np_array(path):
    """Load a multidimensional tiff image as np array"""
    img_array = []
    
    img = imread(path)
    d = img.shape[0]

    for i in range (0,d):
        m_new = imread(path,key=i)
        img_array.append(np.array(m_new, dtype=np.float32))

    # normalise img
    k = np.amax(img_array)
    img_array = np.array(img_array)
    img_array = (img_array)/float(k)
    return img_array

@st.cache
def load_images(imgs, masks, preds, t_idx):
    img = load_img_as_np_array(imgs[t_idx])
    mask = load_img_as_np_array(masks[t_idx])
    pred = load_img_as_np_array(preds[t_idx])

    return img, mask, pred



def draw_bounding_boxes(df, img_bbox, idx):
    
    df_img = df[df["z"] == idx]

    im = Image.fromarray(img_bbox[:, :, idx])
    im = im.convert("RGB")

    draw = ImageDraw.Draw(im)
    colors = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "gray"]

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