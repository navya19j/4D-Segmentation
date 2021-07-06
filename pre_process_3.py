import numpy as np
from PIL import Image, ImageSequence
import sys
import os

from tifffile import *
from tifffile.tifffile import imread,imsave


np.set_printoptions(threshold=sys.maxsize)

def get_binary_image(cell):

    path_dir = os.getcwd() + "/Labeled_Scaled/cell02_EEA1 TagRFP"
    path_fin = os.getcwd() + "/Labeled_vol/cell02_EEA1 TagRFP" 
    for j in range (0,200):
        if (j!=117) and (j!=164) and (j!=66):
            if (int(j/10)==0):
                path_dir_1 = path_dir + "/" + cell + "_T00" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T00" + str(j)
            elif (int(j/100) == 0):
                path_dir_1 = path_dir + "/" + cell + "_T0" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T0" + str(j)
            else: 
                path_dir_1 = path_dir + "/" + cell + "_T" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T" + str(j)

        img = imread(path_dir_1)
        print(img.shape)
        images = []

        print(j)
        
        for i in range (0,img.shape[0]):
            im_new = imread(path_dir_1,key = i)
            images.append(im_new)
        images = np.array(images)
        print(np.shape(images))
        a = 0
        for i in range (0,512,128):
            for j in range (0,128,128):
                for k in range (0,640,128):
                    a+=1
                    vol = images[j:j+128,k:k+128,i:i+128]
                    vol = np.array(vol)
                    if (int(a/10)==0):
                        imsave(path_fin_1+ "_0"+ str(a) +".tif",vol)
                    else:
                        imsave(path_fin_1+ "_"+ str(a) +".tif",vol)
    return 0

cell_1 = "cell02_APPL1 GFP"
cell_2 = "cell02_EEA1 TagRFP"

get_binary_image(cell_2)