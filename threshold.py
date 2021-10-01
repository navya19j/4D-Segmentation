from torch import dtype
import cv2
import numpy as np
import os,sys
from tqdm import tqdm
from skimage.filters import threshold_otsu
#np.set_printoptions(threshold = 100000000)
from tifffile import *
from PIL import Image

def get_filter(image,mask):
    f = image*mask
    return f

def run(predictions):

    path = os.path.join(os.getcwd(),predictions)
    files = (file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))
    all_ims = list(sorted(files))  
    print(f"Thresholding Segmentation Masks for {len(all_ims)} images from {path}")
    img_arr = Image.open(os.path.join(path,all_ims[0]))
    # print(img_arr.shape)
    d = img_arr.n_frames
    del img_arr
    loop = tqdm(all_ims)
    for idx,img in enumerate(loop):
        im = Image.open(os.path.join(path,img))
        final_arr = []
        bounding_coord = {}
        for i in range (0,d):
            loop.set_description(f"Image {idx+1}: Depth {i+1}/{d}")
            new_im = im.seek(i)
            new_im = np.array(im)
            new_im = new_im.astype("uint8")

            if (np.amax(new_im)!=np.amin(new_im)):
                thresh = threshold_otsu(new_im)
                img_otsu = new_im < thresh
                res = get_filter(new_im,img_otsu)
            else:
                res = new_im
            final_arr.append(res)

        final_arr = np.array(final_arr)
        final_arr = final_arr.astype("float32")

        imsave(path+"/"+img,final_arr)
        del final_arr

if __name__ == '__main__':
    predictions = input("Enter directory containing masks to be tracked: ")
    run(predictions)
