import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
import os
import sys
from tqdm import tqdm
from helpers import *
from scipy.optimize import linear_sum_assignment as linear_assignment

def run(predictions,iou_threshold):

    """
        This function returns the maximum sized (area) bounding box of some object. 
        Two boxes are said to be enclosing the same object by the "iou_threshold" measure
        Filters out unwanted/multiple boxes.

        args:
            predictions : Folder Name containing the predicted images
            iou_threshold : Minimum IOU threshold

    """
    start_directory = os.path.join(os.getcwd(),predictions,"bounding_box")
    final_directory = os.path.join(os.getcwd(),predictions,"complete_bounding_box")
    
    print(f"Tracking across z-direction")

    loop = tqdm(list(sorted(os.listdir(start_directory))))
    for idx,files in enumerate(loop):

        file_path = os.path.join(start_directory,files)

        file = open(file_path,"r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)

        temp = conv_dict_to_class(dictionary)
        
        if (temp):
            start = list(temp.keys())[0]
            tracked = temp[start]
            n = 0
            for i in temp:
                n+=1
                loop.set_description(f"Image {idx+1}: Tracking at Depth z={i}/{max(temp)}")
                tracked = assign_detection_to_tracker(tracked,temp[i],iou_threshold,[])

            output = {0:[]}

            for i in tracked:
                f = []
                f.append(i.x)
                f.append(i.y)
                f.append(i.w)
                f.append(i.h)

                output[0].append(f)
        else:
            output = {0:[]}

        filename = files[0:len(files)-4]
        sample = open(os.path.join(final_directory, filename+"_bb1.txt"),"w")
        print(output,file=sample)

        file.close()

def assign_detection_to_tracker(trackers,detections,iou_threshold,final):
    
    iou_matrix = np.zeros((len(detections),len(trackers)))

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = IOU(det,trk)

    matched_indices = linear_assignment(-iou_matrix)

    for d,det in enumerate(detections):
        if d not in matched_indices[0][:]:
            final.append(detections[d])

    for t,trk in enumerate(trackers):
        if t not in matched_indices[1][:]:
            final.append(trackers[t])

    for i in range (0, len(matched_indices[0])):
        tracked = trackers[matched_indices[1][i]]
        detected = detections[matched_indices[0][i]]
        if iou_matrix[matched_indices[0][i],matched_indices[1][i]] >= iou_threshold:
            if (tracked.area()>detected.area()):
                final.append(tracked)

            else:
                final.append(detected)
        else:
            final.append(detected)
            final.append(tracked)

    return final
            

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
