import math
import sys

import cv2
import numpy as np
from PIL import Image

# from tifffile import imsave
from pre_process_original import get_binary_image

np.set_printoptions(threshold=sys.maxsize)


# def conv_to_bw(arr):

#     arr = np.where(arr > 0, 1, 0)
#     arr = arr * 255

#     return arr


def get_final(path_dir_1: str, mask: bool = False, downsample: bool = True):
    """
    Load the image from disk, and optionally resize it
    Args:
        path_dir_1: path to the image
        mask: unused
        downsample: downsample the images to 128x128
    Returns:
        img: np.array of image
    """

    img = Image.open(path_dir_1)
    output = get_binary_image(img)
    images = output[0]

    if downsample:

        d = output[1]
        h = output[2]
        w = output[3]
        max_dim = output[4]

        start = math.ceil(float((max_dim - d) / 2))
        if start != 0:
            start = start + 1
        end = start + 128

        final = []

        for i in range(start, end):
            img = images[i]
            res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            final.append(np.array(res))

        out = np.array(final)

        return np.array(out)
    else:
        return np.array(images)
