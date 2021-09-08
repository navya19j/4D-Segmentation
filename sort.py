import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
import os
from helpers import *
from scipy.optimize import linear_sum_assignment as linear_assignment


# print(dictionary)

def run():
    start_directory = os.path.join(os.getcwd(),"Labeled/cell02_EEA1 TagRFP contours/bounding_box")
    final_directory = os.path.join(os.getcwd(),"Labeled/cell02_EEA1 TagRFP contours/complete_bounding_box")
    
    for files in list(os.listdir(start_directory)):

        file_path = os.path.join(start_directory,files)

        file = open(file_path,"r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)

        temp = conv_dict_to_class(dictionary)

        start = list(temp.keys())[0]
        tracked = temp[start]
        n = 0

        for i in temp:
            n+=1
            tracked = assign_detection_to_tracker(tracked,temp[i],0.2,[])

        # print(tracked)

        output = {0:[]}

        for i in tracked:
            f = []
            f.append(i.x)
            f.append(i.y)
            f.append(i.w)
            f.append(i.h)

            output[0].append(f)

        filename = files[0:len(files)-8]
        print(filename)
        sample = open(os.path.join(final_directory, filename+"_bb1.txt"),"w")
        print(output,file=sample)
        print("Done")

        file.close()

def assign_detection_to_tracker(trackers,detections,iou_threshold,final):
    
    iou_matrix = np.zeros((len(detections),len(trackers)))

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = IOU(det,trk)

    matched_indices = linear_assignment(-iou_matrix)
    # print(matched_indices)

    for d,det in enumerate(detections):
        if d not in matched_indices[0][:]:
            final.append(detections[d])

    for t,trk in enumerate(trackers):
        if t not in matched_indices[1][:]:
            final.append(trackers[t])

    for i in range (0, len(matched_indices[0])):
        tracked = trackers[matched_indices[1][i]]
        detected = detections[matched_indices[0][i]]
        if iou_matrix[matched_indices[0][i],matched_indices[1]][i] >= iou_threshold:
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
        #print(dictionary[i])
    return dictionary

if __name__ == "__main__":
    run()
