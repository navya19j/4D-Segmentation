import os
import numpy as np
import torch
from PIL import Image
from tifffile import imsave
from pre_process_original import *
from pre_process_2 import *
from tifffile import *
from tifffile.tifffile import imread

np.set_printoptions(threshold = 100000000)
torch.set_printoptions(profile="full")

#np.set_printoptions(threshold=np.inf)

class Dataset_Test(torch.utils.data.Dataset):
    def __init__(self,root,data,cellname):
        self.root = root
        # self.transforms = transforms
        self.data = data
        self.cellname = cellname
        self.imgs = list(sorted(os.listdir(os.path.join(root, data,cellname))))
        
    def __getitem__(self,idx):
           
        img_path = os.path.join(self.root,self.data,self.cellname, self.imgs[idx])
        img = get_final(img_path,False)
    
        d = img.shape[0]
        img_array = []
        for i in range (0,d):
            m_new = img[i]
            img_array.append(np.array(m_new, dtype=np.float32))

        k = np.amax(img_array)
        img_array = np.array(img_array)
        img_array = (img_array)/float(k)

        img_array = np.expand_dims(img_array,axis=0)

        return img_array

    def __len__(self):
        return len(self.imgs)
