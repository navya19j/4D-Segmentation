import numpy as np
from PIL import Image
import sys
import os
import math
from tifffile import imsave


np.set_printoptions(threshold=sys.maxsize)

def get_binary_image(img):
 
    # for j in range (0,200):
    #     if (j!=117) and (j!=164): #117
    #         path_dir = os.getcwd() + "/Labeled/cell02_EEA1 TagRFP_binary"
    #         path_fin = os.getcwd() + "/Labeled_Scaled/cell02_EEA1 TagRFP"
    #         if (int(j/10)==0):
    #             path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T00" + str(j) + ".tif"
    #             path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T00" + str(j) + ".tif"
    #             #path_fin_3 = path_fin_2 + "/" + cell + "_T00" + str(j) + ".tif"
    #         elif (int(j/100) == 0):
    #             path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T0" + str(j) + ".tif"
    #             path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T0" + str(j) + ".tif"
    #             #path_fin_3 = path_fin_2 + "/" + cell + "_T0" + str(j) + ".tif"
    #         else:
    #             path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T" + str(j) + ".tif"
    #             path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T" + str(j) + ".tif"
    #             #path_fin_3 = path_fin_2 + "/" + cell + "_T" + str(j) + ".tif"
        
    #     img = Image.open(path_dir_1)
        images = []
        for i in range (0,img.n_frames):
            res = img.seek(i)
            images.append(np.array(img,dtype=np.float32))
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