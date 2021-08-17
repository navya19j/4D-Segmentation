import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
from helper3D import *
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

file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T000_3Dboxes.txt","r")
contents = file.read()
T0 = ast.literal_eval(contents)

file.close()

file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T001_3Dboxes.txt","r")
contents = file.read()
T1 = ast.literal_eval(contents)

file.close()

T0 = conv_dict_to_class(T0)
T1 = conv_dict_to_class(T1)

def assign_detection_to_tracker(detections, trackers, iou_threshold = 0.4):

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

    

# print(assign_detection_to_tracker(T0[0],T1[0],iou_threshold=0.4))

match_ind = assign_detection_to_tracker(T0[0],T1[0],iou_threshold=0.4)[0]

# sample = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T001_track.txt","w")

# print(match_ind,file=sample)
# print("Done")

def expand(match_ind):

    for i in range (0,len(match_ind)):

        t0 = match_ind[i][0]
        t1 = match_ind[i][1]
        match_ind[i] = []

        match_ind[i].append(T0[0][t0])
        match_ind[i].append(T1[0][t1])

    return match_ind

print(expand(match_ind))


