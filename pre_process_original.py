import numpy as np
from PIL import Image
import sys
import os
from tifffile import imsave


np.set_printoptions(threshold=sys.maxsize)

def get_binary_image():

    
 
    for j in range (0,200):
        if (j!=117) and (j!=164): #117
            path_dir = os.getcwd() + "/Labeled/cell02_EEA1 TagRFP_binary"
            path_fin = os.getcwd() + "/Labeled_Scaled/cell02_EEA1 TagRFP"
            if (int(j/10)==0):
                path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T00" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T00" + str(j) + ".tif"
                #path_fin_3 = path_fin_2 + "/" + cell + "_T00" + str(j) + ".tif"
            elif (int(j/100) == 0):
                path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T0" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T0" + str(j) + ".tif"
                #path_fin_3 = path_fin_2 + "/" + cell + "_T0" + str(j) + ".tif"
            else:
                path_dir_1 = path_dir + "/" + "cell02_EEA1 TagRFP" + "_T" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + "cell02_EEA1 TagRFP" + "_T" + str(j) + ".tif"
                #path_fin_3 = path_fin_2 + "/" + cell + "_T" + str(j) + ".tif"
        
        img = Image.open(path_dir_1)
        images = []
        for i in range (0,img.n_frames):
            img.seek(i)
            images.append(np.array(img,dtype=np.float32))
        images = np.array(images,dtype=np.float32)
        print(j)
        print(np.shape(images))
        
        #image.shape = 608,608,608
        images = np.pad(images,((264,),(0,),(0,)),'wrap')
        images = np.pad(images,((0,),(0,),(104,)),'wrap')
        print(np.shape(images))
        # imsave(path_fin_1,images)
        #im = Image.open(path_fin_1)
        #x = im.resize((256,256),resample =Image.NEAREST)
        #new_im = np.zeros((128,128,128))

        #Z-Slice
        # for i in range (0,im.n_frames):
        #     im.seek(i)
        #     if ((i>24) and (i<144)):
        #         x = im.resize((128,128),resample =Image.NEAREST)
        #         x_im = np.array(x)
        #         new_im[i-24] = x
                
        # new_im = np.array(new_im)
        # imsave(path_fin_3,new_im)
        # print(np.shape(new_im))
            #images = np.array(images)

            # #Y-Slice
            # for k in range (103,504):
            #     y = images[:,:,k]
            #     im = Image.fromarray(y)
            #     z =  im.resize((512,512),resample =Image.NEAREST)


            # #X-Slice
            # for l in range (0,608):
            #     o = images[:,l,:]
            #     im = Image.fromarray(o)
            #     p =  im.resize((512,512),resample =Image.NEAREST)

            
            #print(np.shape(images))
            #np.pad(images, ())
            #print(path_fin)
            #print(path_fin_1)


get_binary_image()