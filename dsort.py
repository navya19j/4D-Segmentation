import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
import cv2
from helper3D import *
from tifffile import *
import os
from scipy.optimize import linear_sum_assignment as linear_assignment

def conv_dict_to_class(dictionary):

    
    for j in range (0,len(dictionary[0])):
        x_1 = dictionary[0][j][0]
        y_1 = dictionary[0][j][1]
        z_1 = dictionary[0][j][2]
        w_1 = dictionary[0][j][3]
        h_1 = dictionary[0][j][4]
        d_1 = dictionary[0][j][5]

        dictionary[0][j] = Box(x_1,y_1,z_1,w_1,h_1,d_1)
        
    return dictionary

def assign_detection_to_tracker(detections, trackers, iou_threshold = 0.3):

    iou_matrix = np.zeros((len(detections),len(trackers)))

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = IOU(det,trk)

    matched_indices = linear_assignment(-iou_matrix)
    # print(matched_indices)
    unmatched_detections = []

    for d,det in enumerate(detections):
        if d not in matched_indices[0][:]:
            unmatched_detections.append(d)

    unmatched_trackers = []

    for t,trk in enumerate(trackers):
        if t not in matched_indices[1][:]:
            unmatched_trackers.append(t)

    matches = []

    for i in range (0, len(matched_indices[0])):
        if iou_matrix[matched_indices[0][i],matched_indices[1]][i] < iou_threshold:
            unmatched_detections.append(matched_indices[0][i])
            unmatched_trackers.append(matched_indices[1][i])

        else:
            matches.append((matched_indices[0][i],matched_indices[1][i]))

    # if len(matches) == 0:
    #     matches = np.empty((0,2),dtype = int)
    # else:
    #     matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def ind_to_box(matches,tracker):

    out = []

    for i in range (0,len(matches)):
        out.append(tracker[i])

    return out

def expand(match_ind,T0,T1):

    mat_ind = match_ind.copy()
    for i in range (0,len(mat_ind)):

        t0 = mat_ind[i][0]
        t1 = mat_ind[i][1]
        mat_ind[i] = []

        mat_ind[i].append(T0[0][t0])
        mat_ind[i].append(T1[0][t1])

    return mat_ind

# print(expand(match_ind))

def draw_box(img0,j,current,col):

    new_im = imread(img0,key = j)
    new_im = np.array(new_im)
    new_im_n = np.zeros((608,400,1))
    new_im_n[:,:,0] = new_im
    new_im_n = np.uint8(new_im_n)
    res = new_im_n.copy()
    res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
    
    x = current.x
    y = current.y
    w = current.w
    h = current.h

    cv2.rectangle(res,(x,y),(x+w,y+h),col,1)
    cv2.imshow("output",res)
    cv2.waitKey(0)
    
def conv_box_to_list(box1):
    return [box1.x,box1.y,box1.z,box1.w,box1.h,box1.d]

def run():

    file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T000_3Dboxes.txt","r")
    contents = file.read()
    T0 = ast.literal_eval(contents)

    file.close()

    file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T001_3Dboxes.txt","r")
    contents = file.read()
    T1 = ast.literal_eval(contents)

    file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_3Dboxes.txt","r")
    contents = file.read()
    T2 = ast.literal_eval(contents)

    file.close()

    T0 = conv_dict_to_class(T0)
    T1 = conv_dict_to_class(T1)
    T2 = conv_dict_to_class(T2)

    match_ind_1 = assign_detection_to_tracker(T0[0],T1[0],iou_threshold=0.3)[0]
    detect = ind_to_box(match_ind_1,T1[0])

    match_ind_2 = assign_detection_to_tracker(detect,T2[0],iou_threshold=0.3)[0]

    box_match_1 = expand(match_ind_1,T0,T1)
    box_match_2 = expand(match_ind_2,T1,T2)

    path = os.getcwd() + "/Labeled/cell02_EEA1 TagRFP_binary"
    img0 = path + "/" + "cell02_EEA1 TagRFP_T000.tif"
    img1 = path + "/" + "cell02_EEA1 TagRFP_T001.tif"
    img2 = path + "/" + "cell02_EEA1 TagRFP_T002.tif"

    # two_image(img0,img1,box_match_1)
    # two_image(img1,img2,box_match_2)

    three_image(img0,img1,img2,box_match_1,box_match_2)

def two_image(img0,img1,box_match_1):

    for i in range (0,len(box_match_1)):

        current = box_match_1[i][0]
        future = box_match_1[i][1]

        maxstart = min(current.z,future.z)
        maxend = max(current.z+current.d,future.z+future.d)
        
        for j in range (maxstart,maxend):

            if (j<current.z+current.d+1 and j>= current.z):
                draw_box(img0,j,current,(0,255,0))

            if (j<future.z+future.d+1 and j>= future.z):
                draw_box(img1,j,future,(0,0,255))
        
        print("Done")

def three_image(img0,img1,img2,box_match_1,box_match_2):

    for i in range (0,len(box_match_1)):

        current = box_match_1[i][0]
        future = box_match_1[i][1]

        maxstart = min(current.z,future.z)
        maxend = max(current.z+current.d,future.z+future.d)
        next_fut = '#'

        for k in range (0,len(box_match_2)):

            if (conv_box_to_list(box_match_2[k][0])==conv_box_to_list(future)):
                next_fut = box_match_2[k][1]
                break

        if (next_fut!='#'):
            maxstart = min(maxstart,next_fut.z)
            maxend = max(maxend,next_fut.z+next_fut.d)

        for j in range (maxstart,maxend):

            if (j<current.z+current.d+1 and j>= current.z):
                draw_box(img0,j,current,(0,255,0))

            if (j<future.z+future.d+1 and j>= future.z):
                draw_box(img1,j,future,(0,0,255))

            if (next_fut!='#'):
                if (j<next_fut.z+next_fut.d+1 and j>= next_fut.z):
                    draw_box(img2,j,next_fut,(255,0,0))
        
        print("Done")

run()




