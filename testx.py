import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tifffile import imsave
from tifffile import *
from numpy import fabs
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch.optim as optimizer
from unet import *
from dataset import *
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(device)
    #print(t,r,a)
if (device=="cuda"):
    torch.cuda.empty_cache()
path = os.getcwd()
model = UNet(in_channels=1,out_channels=1)
model.to(device)
def save_prediction(arr):
    final_arr = np.zeros(128,512,640)
    arr_n = []
    ind = arr[0].rfind("_")
    mask_path = (arr[0])[0:ind]
    for im in arr:
        path = os.getcwd() + "/Data_vol/cell02_EEA1 TagRFP/" + im
        img = imread(path)
        d = img.shape[0]
        load_checkpoint(torch.load("new.pth.tar"),model)

        h_mask,w_mask = img.shape[1],img.shape[2]
        img_array = np.zeros([1,1,d,w_mask,h_mask])

        #img_array = np.zeros([1,d,w_mask,h_mask])
        #print(mask_path)
        for i in range (0,d):
            im_new = imread(path,key = i)
            img_array[:,:,i,:,:] = np.array(im_new, dtype=np.float32)
        k = np.amax(img_array)
        img_array = img_array*255/float(k)

        x = torch.from_numpy(img_array).float().to(device)
        x = model(x)
        pred = torch.sigmoid(x)
        pred = (pred > 0.5).float().detach()

        pred.squeeze(0)
        pred.squeeze(0)
        pred = pred*255
        
        arr_n.append(np.array(pred)) 

    a = 0
    for i in range (0,4):
        for j in range (0,5):
            final_arr[:,j:128*(j+1),i:128*(i+1)] = arr_n[a]
            a+=1
    
    final_arr = final_arr[24:104,16:624,56:456]
    final_arr = np.array(final_arr)

    imsave(os.getcwd() + "/Data_vol/cell02_EEA1 TagRFP/"+mask_path + ,final_arr)
            # mask_actual.seek(i)
            # y = mask_actual.resize((608,608),resample =Image.NEAREST)
            # y_im = np.array(y)[:,103:503]
            # mask_array[i,:,:] = y_im
            
#         imsave(os.getcwd()+"/checkpred.tif",img_array)
#         #imsave(path2,mask_array)  
#         print("Saved")


# imsave(os.getcwd()+"/checkpred1.tif",pred.detach().cpu().numpy())
#         path1 = os.getcwd()+"/checkpred1.tif"
#         mask_predicted = Image.open(path1)
#         print(mask_predicted.size)
#         d = 51
#         img_array = np.zeros([51,800,567])
#         for i in range (40,91):
#             mask_predicted.seek(i)
#             x = mask_predicted.resize((800,800),resample =Image.NEAREST)
#             x_im = np.array(x)[:,116:683]
#             img_array[i-40,:,:] = x_im


all_img = list(sorted(os.listdir(os.path.join(root,"Data_vol/cell02_EEA1 TagRFP"))))
for i in range (0,199):
    arr_new = []
    for img in all_img:
        if (int(i/10)==0):
            i_new = "00" + str(i)
        elif (int(idx/100) == 0):
            i_new = "0" + str(i)
        else:
            i_new =  str(i)

        if ("T"+i_new) in img:
            arr_new.append(img)
    save_prediction(arr_new)
    
            