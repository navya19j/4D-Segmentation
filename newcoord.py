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
from tqdm import tqdm
import ast
from helpers import *
from scipy.optimize import linear_sum_assignment as linear_assignment




def run(predictions,min_vol):
    """
        Returns all the 3D bounding boxes in the image file.

        args:
            predictions : Folder Name containing the predicted images
            min_vol : Minimum Volume of the bounding cube to be classified as a valid bounding box for detected object

    """

    print(f"Generating 3D Bounding Boxes")
    start_directory_1 = os.path.join(os.getcwd(),predictions,"bounding_box")
    start_directory_2 = os.path.join(os.getcwd(),predictions,"complete_bounding_box")

    final_directory = os.path.join(os.getcwd(),predictions,"3D_Box")
    x = 0
    loop = tqdm(list(sorted(os.listdir(start_directory_1))))
    for idx,files in enumerate(loop):
        loop.set_description(f"Image {idx+1}: Loading Tracks")
        file_path = os.path.join(start_directory_1,files)

        file = open(file_path,"r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file_path_2 = os.path.join(start_directory_2, files[0:len(files)-4] +"_bb1.txt")


        file = open(file_path_2,"r")
        contents = file.read()
        bbox = ast.literal_eval(contents)

        path = os.path.join(os.getcwd(),predictions)

        imgbox = conv_dict_to_class(dictionary)
        bbox = conv_dict_to_class(bbox)

        img = files[0:len(files)-4] + ".tif"

        finalbbox = {}
        isvisited = {}
        final_arr = []
        depth = {}
        start = {}

        if (x==0):
            path_img = os.path.join(path,img)
            img_arr = imread(path_img)
            d = img_arr.shape[0]

        for i in range (0,d):
            loop.set_description(f"Image {idx+1}: Depth {i+1}/{d}")
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
                        bb = [coord.x,coord.y,coord.w,coord.h]

                        if (coord in isvisited):
                            if (isvisited[coord]==1):
                                depth[coord] += 1
                
                        else:
                            isvisited[coord] = 1
                            start[coord] = i+1
                            depth[coord] = 1
                            
                        finalbbox[i+1].append(bb)
        allbox = {0:[]}

        for i in start:
            temp = [i.x,i.y,i.w,i.h]

            temp.insert(2,start[i])
            temp.append(depth[i])
            vol = temp[-1]*temp[-2]*temp[-3]
            if (vol>=int(min_vol)):
                allbox[0].append(temp)

        sample = open(os.path.join(final_directory, files[0:len(files)-4] +"_3Dboxes.txt"),"w")

        print(allbox,file=sample)
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

    run(sys.argv[2])
