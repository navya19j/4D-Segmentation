import numpy as np
from PIL import Image
import sys
import os
import math
from tifffile import imsave


np.set_printoptions(threshold=sys.maxsize)

def get_binary_image(img):
 
        images = []
        for i in range (0,img.n_frames):
            res = img.seek(i)
            # print(np.array(img,dtype=np.float32).shape)
            images.append(np.array(img,dtype=np.float32))
            # print(images)
        images = np.array(images,dtype=np.float32)
        # 80,608,400
        #image.shape = 608,608,608
        d = images.shape[0]
        h = images.shape[1]
        w = images.shape[2]

        max_dim = max(max(h,w),d)
        # print(max_dim)

        d_up = int(math.ceil(float((max_dim-d)/2)))
        d_down = int(math.floor(float((max_dim-d)/2)))
        # print(d_up,d_down)

        w_up = int(math.ceil(float((max_dim-w)/2)))
        w_down = int(math.floor(float((max_dim-w)/2)))
        # print(w_up,w_down)

        h_up = int(math.ceil(float((max_dim-h)/2)))
        h_down = int(math.floor(float((max_dim-h)/2)))
        # print(h_up,h_down)

        images = np.pad(images,((d_up,d_down),(0,0),(0,0)),'wrap')
        images = np.pad(images,((0,0),(0,0),(w_up,w_down)),'wrap')
        images = np.pad(images,((0,0),(h_up,h_down),(0,0)),'wrap')

        # print(images.shape)

        return (images,d,h,w,max_dim)