import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
from helpers import *
from scipy.optimize import linear_sum_assignment as linear_assignment


file = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002.tif.txt","r")
contents = file.read()
dictionary = ast.literal_eval(contents)

file.close()
# print(dictionary)

def assign_detection_to_tracker(tracker,detection,iou_threshold,final):
    
    for i in range (0,len(tracker)):
        visit = 1
        for j in range (0,len(detection)):
            tracked = tracker[i]
            detected = detection[j]

            if (IOU(tracked,detected)>=iou_threshold):
                visit = 0
                if tracked.area() >= detected.area():
                    detection[j] = tracked
                    final.append(tracked)
                
                else:
                    try :
                        x = final.index(detected)
                    except:
                        final.append(detected)
        
        if (visit==1):
            detection.append(tracker[i])
            # final.append(tracker[i])

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

#print(conv_dict_to_class(dictionary))
temp = conv_dict_to_class(dictionary)

start = list(temp.keys())[0]
tracked = temp[start]
detected = temp[start]
fin = []
for i in temp:
    assign_detection_to_tracker(temp[i],detected,0.2,fin)

final = []
for i in temp:
    assign_detection_to_tracker(temp[i],detected,0.2,final)

# print(detected)

# sample = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_finalcoord.txt","w")
# for i in detected:
#     print(i.x,file=sample)
#     print(i.y,file=sample)
#     print(i.w,file=sample)
#     print(i.h,file=sample)
#     print(" ",file=sample)

# sample = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_finalcoord2.txt","w")
# for i in final:
#     print(i.x,file=sample)
#     print(i.y,file=sample)
#     print(i.w,file=sample)
#     print(i.h,file=sample)
#     print(" ",file=sample)

output = {0:[]}

for i in detected:
    f = []
    f.append(i.x)
    f.append(i.y)
    f.append(i.w)
    f.append(i.h)

    output[0].append(f)

sample = open("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya/Labeled/cell02_EEA1 TagRFP contours/cell02_EEA1 TagRFP_T002_bb.txt","w")

print(output,file=sample)
print("Done")