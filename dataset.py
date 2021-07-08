import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tifffile import imsave

from tifffile import *
from tifffile.tifffile import imread

np.set_printoptions(threshold = 100000000)
torch.set_printoptions(profile="full")

#np.set_printoptions(threshold=np.inf)

class EndosomeDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms=None):
        self.root = root
        # self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root,"Data_Resized/cell02_EEA1 TagRFP"))))
        self.masks = list(sorted(os.listdir(os.path.join(root,"Labeled_Resized/cell02_EEA1 TagRFP"))))
        
    def __getitem__(self,idx):
        #load images and masks from input and segmented data
        if (int(idx/10)==0):
            idx_new = "00" + str(idx)
        elif (int(idx/100) == 0):
            idx_new = "0" + str(idx)
        else:
            idx_new =  str(idx)
           
        img_path = os.path.join(self.root,"Data_Resized/cell02_EEA1 TagRFP", self.imgs[idx])
        mask_path = os.path.join(self.root,"Labeled_Resized/cell02_EEA1 TagRFP", self.masks[idx])
        #img = Image.open(img_path)
        img = imread(img_path)
        #h,w = np
        # print(img)
        
        d = img.shape[0]
        # print(d)
        mask = imread(mask_path)
        h_mask,w_mask = mask.shape[1],mask.shape[2]
        # print(h_mask)
        # print(w_mask)
        # print("done")
        
        #mask = Image.open(mask_path)
        
        mask_array = []
        img_array = []
        #print(mask_path)
        for i in range (0,d):
            im_new = imread(mask_path,key=i)
            # print(i)
            # print(im_new)
            mask_array.append(np.array(im_new, dtype=np.float32))
        mask_array = np.array(mask_array)
        mask_array = mask_array/float(255)

        for i in range (0,d):
            m_new = imread(img_path,key=i)
            # print(i)
            # print(m_new)
            img_array.append(np.array(m_new, dtype=np.float32))
        k = np.amax(img_array)
        img_array = np.array(img_array)
        img_array = (img_array)/float(k)

        img_array = np.expand_dims(img_array,axis=0)
        mask_array = np.expand_dims(mask_array,axis=0)
        # print(img_array.shape)
        # print(mask_array.shape)
        # if self.transforms is not None:
        #     img_array = self.transforms(img_array)
        #     mask_array = self.transforms(mask_array)

        return img_array,mask_array

    def __len__(self):
        return len(self.imgs)

# print(EndosomeDataset("C:/Users/navya/Desktop/Stuff/IIT/4D segmentation/Data_Navya")[0])