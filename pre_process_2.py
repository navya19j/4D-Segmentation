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


def get_final(path_dir_1,mask,truth):
    
    if (truth == "Y"):
        img = Image.open(path_dir_1)
        
        output = get_binary_image(img)
        images = output[0]
        # d = output[1]
        # h = output[2]
        # w = output[3]
        # max_dim = output[4]

        # start = math.ceil(float((max_dim-d)/2))
        # if (start!=0):
        #     start = start+1
        # end = start+128
        
        # final = []

        # for i in range (start,end):
        #     img = images[i]
        #     res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        #     final.append(np.array(res))

        # out = np.array(final)
        # print(out.shape)
    
        return np.array(images)
    else:
        img = Image.open(path_dir_1)
        
        output = get_binary_image(img)
        images = output[0]
        d = output[1]
        h = output[2]
        w = output[3]
        max_dim = output[4]

        start = math.ceil(float((max_dim-d)/2))
        if (start!=0):
            start = start+1
        end = start+128
        
        final = []

        for i in range (start,end):
            img = images[i]
            res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            final.append(np.array(res))

        out = np.array(final)
        # print(out.shape)
    
        return np.array(out)