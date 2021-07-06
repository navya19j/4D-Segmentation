import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tifffile import imsave

from tifffile import *
from tifffile.tifffile import imread
import torchio as tio

np.set_printoptions(threshold = 100000000)
torch.set_printoptions(profile="full")

#np.set_printoptions(threshold=np.inf)

class EndosomeDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
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
        #print(img_path)
        #print(mask_path)
        #d = mask.shape[0]
        #print("done")
        
        #mask = Image.open(mask_path)
        mask = imread(mask_path)
        #h_mask,w_mask = mask.shape[1],mask.shape[2]
        mask_array = np.zeros([1,128,128,128])

        img_array = np.zeros([1,128,128,128])
        #print(mask_path)
        for i in range (0,128):
            im_new = imread(mask_path,key=i)
            mask_array[:,i,:,:] = np.array(im_new, dtype=np.float32)
        mask_array = mask_array//float(255)

        for i in range (0,128):
            m_new = imread(img_path,key=i)
            #print(im_new)
            img_array[:,i,:,:] = np.array(m_new, dtype=np.float32)
        k = np.amax(img_array)
        img_array = img_array*255/float(k)

        ##print(img_array)
        #print(mask_array)
        if self.transforms is not None:
            img_array = self.transforms(img_array)
            mask_array = self.transforms(mask_array)

        return img_array,mask_array

    def __len__(self):
        return len(self.imgs)
