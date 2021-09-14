import numpy as np
import sys
from tifffile import imsave
from pre_process_original import *
import cv2

np.set_printoptions(threshold=sys.maxsize)


def conv_to_bw(arr):

    arr = np.where(arr > 0, 1, 0)
    arr = arr*255
    
    return arr


def get_final(path_dir_1,mask):
    
    img = Image.open(path_dir_1)
    images = get_binary_image(img)
    final = []

    for i in range (264,392):
        img = images[i]
        res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        final.append(np.array(res))

    out = np.array(final)
 
    return out