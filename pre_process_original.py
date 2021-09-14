import numpy as np
from PIL import Image
import sys
import os
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
        images = np.pad(images,((264,),(0,),(0,)),'wrap')
        images = np.pad(images,((0,),(0,),(104,)),'wrap')

        return images