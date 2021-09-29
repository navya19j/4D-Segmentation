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

def run(predictions):

    start_directory_1 = os.path.join(os.getcwd(),predictions,"bounding_box")
    start_directory_2 = os.path.join(os.getcwd(),predictions,"complete_bounding_box")

    final_directory = os.path.join(os.getcwd(),predictions,"3D_Box")
    x = 0

    for files in list(os.listdir(start_directory_1)):

        file_path = os.path.join(start_directory_1,files)

        file = open(file_path,"r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file_path_2 = os.path.join(start_directory_2, files[0:len(files)-4] +"_bb1.txt")


        file = open(file_path_2,"r")
        contents = file.read()
        bbox = ast.literal_eval(contents)

        path = os.path.join(os.getcwd(),predictions)
        # path_final = os.getcwd() + "Labeled/cell02_EEA1 TagRFP contours/temp"

        imgbox = conv_dict_to_class(dictionary)
        bbox = conv_dict_to_class(bbox)

        img = files[0:len(files)-4] + ".tif"
        # imgs = files[0:len(files)-8] + "f1" ".tif"
        finalbbox = {}
        isvisited = {}
        final_arr = []
        depth = {}
        start = {}

        if (x==0):
            path_img = os.path.join(path,img)
            img_arr = imread(path_img)
            # print(img_arr.shape)
            d = img_arr.shape[0]

        for i in range (0,d):
            new_im = imread(os.path.join(path,img),key = i)
            # new_im = imread(path+"/"+img,key = i)
            new_im = np.array(new_im)
            w = new_im.shape[0]
            h = new_im.shape[1]
            new_im_n = np.zeros((w,h,1))
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

        # imsave(path_final+"/"+imgs,final_arr)
        # print("Done")
        # print(finalbbox)
        allbox = {0:[]}

        for i in start:
            temp = [i.x,i.y,i.w,i.h]

            temp.insert(2,start[i])
            temp.append(depth[i])
            allbox[0].append(temp)

        sample = open(os.path.join(final_directory, files[0:len(files)-4] +"_3Dboxes.txt"),"w")

        print(allbox,file=sample)
        print("Done")
        x+=1

def conv_dict_to_class(dictionary):

    for i in dictionary:
        for j in range (0,len(dictionary[i])):
            x_1 = dictionary[i][j][0]
            y_1 = dictionary[i][j][1]
            w_1 = dictionary[i][j][2]
            h_1 = dictionary[i][j][3]

            dictionary[i][j] = Box(x_1,y_1,w_1,h_1)
        
    return dictionary

if __name__ == "__main__":
    # predictions = input("Enter directory containing masks to be tracked: ")
    run(sys.argv[2])