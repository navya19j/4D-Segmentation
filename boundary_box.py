from torch import dtype
import cv2
import numpy as np
import os,sys
#np.set_printoptions(threshold = 100000000)
from tifffile import *
from PIL import Image,ImageDraw



def create_bound_box(predictions):
    path = os.path.join(os.getcwd(),predictions)
    path_final = os.path.join(os.getcwd(),predictions ,"bounding_box")
    # all_ims = list(sorted(os.listdir(path)))
    files = (file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))
    all_ims = list(sorted(files))
    for img in all_ims:
    
        final_arr = []
        bounding_coord = {}
        
        for i in range (0,80):
            new_im = imread(os.path.join(path,img),key = i)
            new_im = np.array(new_im)
            new_im_n = np.zeros((608,400,1))
            new_im_n[:,:,0] = new_im
            new_im_n = np.uint8(new_im_n)
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
        sample = open(os.path.join(path_final, img[:len(img)-4]+".txt"),"w")
        # sample = open(path_final+"/"+img[:len(img)-4]+".txt","w")
        print(bounding_coord, file = sample)
        sample.close()
        
    return (bounding_coord)


if __name__ == '__main__':
    # predictions = input("Enter directory containing masks to be tracked: ")
    create_bound_box(sys.argv[2])