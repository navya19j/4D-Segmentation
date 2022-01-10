from torch import dtype
import cv2
import numpy as np
import os,sys
from tqdm import tqdm
from tifffile import *
from PIL import Image,ImageDraw

def create_bound_box(predictions,min_area):

    """"
        Uses the predicted masks to find a straight bounding rectangle for the detected contours in the image.

        args:
            predictions : Folder Name containing the predicted images
            min_area : Minimum Area of the bounding rectangle to be classified as a valid bounding box for detected object
    """
    
    path = os.path.join(os.getcwd(),predictions)
    path_final = os.path.join(os.getcwd(),predictions,"bounding_box")
    files = (file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))
    all_ims = list(sorted(files))  
    img_arr = imread(os.path.join(path,all_ims[0]))
    d = img_arr.shape[0]

    print(f"Generating 2D Bounding Boxes for {len(all_ims)} images from {path}")
    
    loop = tqdm(all_ims)

    for idx,img in enumerate(loop):
    
        try:
            final_arr = []
            bounding_coord = {}
            
            for i in range (0,d):
                loop.set_description(f"Image {idx+1}: Depth {i+1}/{d}")
                new_im = imread(os.path.join(path,img),key = i)
                new_im = np.array(new_im)
                w = new_im.shape[0]
                h = new_im.shape[1]
                new_im_n = np.zeros((w,h,1))
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
                    if (w*h>=int(min_area)):
                        if (to_find in bounding_coord):
                            new = []
                            for ind in bounding_coord[i+1]:
                                new.append(ind)
                            new.append([x,y,w,h])
                            bounding_coord[i+1] = new
                        else:
                            bounding_coord[i+1] = [[x,y,w,h]]
                final_arr.append(res)

            sample = open(os.path.join(path_final, img[:len(img)-4]+".txt"),"w")
            print(bounding_coord, file = sample)
            sample.close()
        except Exception as e:
            print(e)
            print(f"Unable to load {img}")

        
    return (bounding_coord)


if __name__ == '__main__':
    create_bound_box(sys.argv[2])
