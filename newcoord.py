from torch import dtype
import cv2
import numpy as np
import os,sys
#np.set_printoptions(threshold = 100000000)
from tifffile import *
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
from helpers import *
from scipy.optimize import linear_sum_assignment as linear_assignment

file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002.tif.txt","r")
contents = file.read()
dictionary = ast.literal_eval(contents)

file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_bb1.txt","r")
contents = file.read()
bbox = ast.literal_eval(contents)

path = "C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/" + "Labeled/cell02_EEA1 TagRFP_binary"
path_final = "C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/" + "Labeled/cell02_EEA1 TagRFP contours"
def conv_dict_to_class(dictionary):

    for i in dictionary:
        for j in range (0,len(dictionary[i])):
            x_1 = dictionary[i][j][0]
            y_1 = dictionary[i][j][1]
            w_1 = dictionary[i][j][2]
            h_1 = dictionary[i][j][3]

            dictionary[i][j] = Box(x_1,y_1,w_1,h_1)
        
    return dictionary

imgbox = conv_dict_to_class(dictionary)
bbox = conv_dict_to_class(bbox)

img = "cell02_EEA1 TagRFP_T002.tif"
imgs = "cell02_EEA1 TagRFP_T002f1.tif"
finalbbox = {}
isvisited = {}
final_arr = []
depth = {}
start = {}
for i in range (0,80):

    new_im = imread(path+"/"+img,key = i)
    new_im = np.array(new_im)
    new_im_n = np.zeros((608,400,1))
    new_im_n[:,:,0] = new_im
    new_im_n = np.uint8(new_im_n)
    res = new_im_n.copy()
    res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
    finalbbox[i+1] = []

    if (imgbox.get(i+1)!=None):
        for j in range (0,len(imgbox[i+1])):
            iou = []
            bb_iou = {}
            for k in range (0,len(bbox[0])):
                iou_bb = IOU(imgbox[i+1][j],bbox[0][k])
                iou.append(iou_bb)
                bb_iou[iou_bb] =  bbox[0][k]      
            iou.sort()
            out = iou[-1]
            coord = bb_iou[out]

            if (out >= 0.2):
                cv2.rectangle(res,(coord.x,coord.y),(coord.x+coord.w,coord.y+coord.h),(0,255,0),1)
                bb = [coord.x,coord.y,coord.w,coord.h]

                if (coord in isvisited):
                    if (isvisited[coord]==1):
                        depth[coord] += 1
        
                else:
                    isvisited[coord] = 1
                    start[coord] = i+1
                    depth[coord] = 1
                    
                finalbbox[i+1].append(bb)

    final_arr.append(res)

imsave(path_final+"/"+imgs,final_arr)
print("Done")
# print(finalbbox)
allbox = {0:[]}

for i in start:
    temp = [i.x,i.y,i.w,i.h]

    temp.insert(2,start[i])
    temp.append(depth[i])
    allbox[0].append(temp)

sample = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_3Dboxes.txt","w")

print(allbox,file=sample)
print("Done")




            


