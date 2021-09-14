from torch import dtype
import cv2
import numpy as np
import os,sys
#np.set_printoptions(threshold = 100000000)
from tifffile import *
from PIL import Image,ImageDraw

def create_bound_box():
    path = os.getcwd() + "Labeled/cell02_EEA1 TagRFP_binary"
    path_final = os.getcwd() + "Labeled/cell02_EEA1 TagRFP contours"
    all_ims = list(sorted(os.listdir(path)))
    for img in all_ims:
    
        final_arr = []
        bounding_coord = {}
        
        for i in range (0,80):
            new_im = imread(path+"/"+img,key = i)
            # new_im.seek(i)
            #bounding_coord[] = []
            new_im = np.array(new_im)
            new_im_n = np.zeros((608,400,1))
            new_im_n[:,:,0] = new_im
            new_im_n = np.uint8(new_im_n)
            # cv2.imshow("bb",new_im_n)
            # cv2.waitKey(0)
            # print(new_im_n.shape)
            #new_im = cv2.cvtColor(new_im,cv2.COLOR_GRAY2BGR)
            #print(new_im)4
            #new_im = cv2.cvtColor(new_im,cv2.COLOR_BGR2GRAY)
            #thresh = cv2.threshold(new_im,128,255,cv2.THRESH_BINARY)[1]
            res = new_im_n.copy()
            res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
            edged = cv2.Canny(new_im_n,30,200)

            contours = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            for cntr in contours:
                x,y,w,h = cv2.boundingRect(cntr)
                cent = ((x+w)/2,(y+h)/2)
                to_find = i+1
                if (to_find in bounding_coord):
                    new = []
                    for ind in bounding_coord[i+1]:
                        new.append(ind)
                    new.append([x,y,w,h])
                    bounding_coord[i+1] = new
                else:
                    bounding_coord[i+1] = [[x,y,w,h]]
                cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),1)
            final_arr.append(res)

        sample = open(path_final+"/"+img+".txt","w")
        print(bounding_coord, file = sample)
        sample.close()
        print(img)
        
    return (bounding_coord)

create_bound_box()