import os
import numpy as np
import torch
from PIL import Image
from tifffile import imsave
from pre_process_original import *
from pre_process_2 import *
from tifffile import *
from tifffile.tifffile import imread

np.set_printoptions(threshold=100000000)
torch.set_printoptions(profile="full")

# np.set_printoptions(threshold=np.inf)


class EndosomeDataset(torch.utils.data.Dataset):

    """"
        Class for the Training Dataset

        Test Dataset is structured as follows :
        
        root --> data --> cellname --> all input image files
        root --> label --> cellname --> all masked image files

        args:
            root : Root Directory
            data : Name of directory containing all folders of input data
            label : Name of directory containing all folders of mask data
            cellname : Name of the Directory containing images in either folder
            truth : True if input image is to be reduced in size, otw False
    """

    def __init__(self, root, data, label, cellname, downsample):
        self.root = root
        # self.transforms = transforms
        self.data = data
        self.cellname = cellname
        self.label = label
        self.downsample = downsample
        self.imgs = list(sorted(os.listdir(os.path.join(root, data, cellname))))
        self.masks = list(sorted(os.listdir(os.path.join(root, label, cellname))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.data, self.cellname, self.imgs[idx])
        mask_path = os.path.join(self.root, self.label, self.cellname, self.masks[idx])
        # img = Image.open(img_path)
        # img = Image.open(img_path)
        img = get_final(img_path, False, self.downsample)

        d = img.shape[0]
        # print(d)
        mask = get_final(mask_path, True, self.downsample)

        h_mask, w_mask = mask.shape[1], mask.shape[2]
        # print(h_mask)

        mask_array = []
        img_array = []

        for i in range(0, d):
            im_new = mask[i]
            mask_array.append(np.array(im_new, dtype=np.float32))

        mask_array = np.array(mask_array)
        mask_array = np.array(mask_array)
        mask_array = np.where(mask_array > 0, 1, 0)

        for i in range(0, d):
            m_new = img[i]
            img_array.append(np.array(m_new, dtype=np.float32))

        k = np.amax(img_array)
        img_array = np.array(img_array)
        img_array = (img_array) / float(k)

        img_array = np.expand_dims(img_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=0)

        return img_array, mask_array

    def __len__(self):
        """
            Number of images in the loader
        """
        return len(self.imgs)
