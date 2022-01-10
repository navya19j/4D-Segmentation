import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
import cv2
import sys
from tqdm import tqdm
from helper3D import *
from tifffile import *
from tracker import *
import os
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment as linear_assignment


def conv_dict_to_class(dictionary):

    """
        Create a Box object from dictionary

        args:
            dictionary: given dictionary
    """
    
    for j in range (0,len(dictionary[0])):
        x_1 = dictionary[0][j][0]
        y_1 = dictionary[0][j][1]
        z_1 = dictionary[0][j][2]
        w_1 = dictionary[0][j][3]
        h_1 = dictionary[0][j][4]
        d_1 = dictionary[0][j][5]

        dictionary[0][j] = Box(x_1,y_1,z_1,w_1,h_1,d_1)
        
    return dictionary[0]

def draw_box(img0,current,color,object_id,iter):

    """
        Helper function to visualize bounding boxes

        args:
            img0 : Image File
            current : Box Object
            color : color of bounding box
            object_id : ID of the object detected
            iter : File number (in case of endosome data : time stamp)
    """

    x = current.x
    y = current.y
    z = current.z
    w = current.w
    h = current.h
    d = current.d


    for j in range (z,z+d+1):
        new_im = imread(img0,key = j)
        new_im = np.array(new_im)
        new_im_n = np.zeros((608,400,1))
        new_im_n[:,:,0] = new_im
        new_im_n = np.uint8(new_im_n)
        res = new_im_n.copy()
        res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
        cv2.rectangle(res,(x,y),(x+w,y+h),color,1)
        cv2.putText(res, "(" + str(iter)+ "," +str(object_id)+ ")", (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("output",res)
        cv2.waitKey(0)

    
def conv_box_to_list(box1):
    return [box1.x,box1.y,box1.z,box1.w,box1.h,box1.d]

def get_img_path(idx,path):

    img_id = os.path.join(path,idx)

    return img_id

    
def run(predicitons,iou_threshold):

    """
        Tracks the 3D bounding boxes detected across time.

        args:
            predictions : Folder containing predicted images
            iou_threshold : Minimum IOU threshold
    """

    start_directory_1 = os.path.join(os.getcwd(),predicitons,"3D_Box")
    all_box = []

    tracker = Tracker(iou_threshold)
    img_name = []
    loop = tqdm(list(sorted(os.listdir(start_directory_1))))
    for i,files in enumerate(loop):
        try:
            loop.set_description(f"Loading Tracking Data {i+1}/{len(loop)}")
            file = open(os.path.join(start_directory_1,files),"r")
            contents = file.read()
            T0 = ast.literal_eval(contents)
            all_box.append(T0)
            img_name.append(files[0:len(files)-12]+".tif")
            file.close()
        except:
            print("Ds")

    all_box = [conv_dict_to_class(i) for i in all_box]

    iter = 0
    path = os.path.join(os.getcwd(),predicitons)

    track_map = {}
    object_map = {}

    loop = tqdm(all_box)
    for i,boxes in enumerate(loop):

        loop.set_description(f"Tracking 3D Bounding Box {i+1}/{len(loop)}")
        img_id = img_name[iter]

        objects = tracker.update(boxes)
        # print(objects)

        for (object_id,box) in objects.items():

            if object_map.get(object_id) is None:

                object_map[object_id] = [(iter,conv_box_to_list(box))]

            else:

                object_map[object_id].append((iter,conv_box_to_list(box)))

            if track_map.get(object_id) is not None:
                track_map[object_id].append(iter)

            else:
                track_map[object_id] = [iter]

            # draw_box(get_img_path(img_id,path),box,(0,0,255),object_id,iter)

        iter += 1

    sample_1 = open(os.path.join(path, "track.txt"),"w")
    sample_2 = open(os.path.join(path, "object.txt"),"w")

    print(track_map,file = sample_1)
    print(object_map,file = sample_2)

if __name__ == "__main__":
    # predictions = input("Enter directory containing masks to be tracked: ")
    run("output/cell1/predicted_mask")

