import math
import sys
import cv2
import numpy as np
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

def get_padded_image(img):

    """
        Pads the image across 2 dimensions to create a cube with size as the maximum dimension 
        in the input image

        Args:
            img: PIL Image object
        Returns:
            img, d, h, w, max_dim: np.array of image, depth, height, width, max(d,h,w)
    """

    images = []
    for i in range(0, img.n_frames):
        res = img.seek(i)
        images.append(np.array(img, dtype=np.float32))

    images = np.array(images, dtype=np.float32)

    d = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    max_dim = max(max(h, w), d)

    d_up = int(math.ceil(float((max_dim - d) / 2)))
    d_down = int(math.floor(float((max_dim - d) / 2)))

    w_up = int(math.ceil(float((max_dim - w) / 2)))
    w_down = int(math.floor(float((max_dim - w) / 2)))

    h_up = int(math.ceil(float((max_dim - h) / 2)))
    h_down = int(math.floor(float((max_dim - h) / 2)))

    images = np.pad(images, ((d_up, d_down), (0, 0), (0, 0)), "wrap")
    images = np.pad(images, ((0, 0), (0, 0), (w_up, w_down)), "wrap")
    images = np.pad(images, ((0, 0), (h_up, h_down), (0, 0)), "wrap")

    return (images, d, h, w, max_dim)


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
    output = get_padded_image(img)
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
