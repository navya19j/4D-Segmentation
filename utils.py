import os

import numpy as np
import torch
from numpy.core.fromnumeric import size
from PIL import Image, ImageSequence
from tifffile import imsave
from tqdm import tqdm

from dataset import *
from unet import *

torch.set_printoptions(profile="full")
from dataset_test import *

# test = []

def get_loaders(path,data,label,cellname,part:bool ,truth,bat_size):
    
    dataset_all = EndosomeDataset(path,data,label,cellname,truth)
    torch.manual_seed(5)

    indices = torch.randperm(len(dataset_all)).tolist()
    # if (117 in indices):
    #     indices.remove(117)
    # if (66 in indices):
    #     indices.remove(66)
    
    if part:
        length = len(indices)
        idx = int(-0.2*length)
        # print(idx)
    
        dataset_train = torch.utils.data.Subset(dataset_all, indices[:idx])

        dataset_test = torch.utils.data.Subset(dataset_all, indices[idx:])
        print(f"Chose {-idx} Test Images")
        # print(indices[idx:])
        test = indices[idx:]

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=bat_size, num_workers=0, shuffle=True,pin_memory=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,num_workers=0, shuffle=False,pin_memory=True)

        return data_loader_train,data_loader_test
    else:

        dataset_train = torch.utils.data.Subset(dataset_all, indices)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=bat_size, num_workers=0, shuffle=True,pin_memory=True)

        return data_loader_train

def get_loaders_test(path,data,cellname,truth):
    
    dataset_all = Dataset_Test(path,data,cellname,truth)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_all, batch_size=1,num_workers=0, shuffle=False,pin_memory=True)

    return data_loader_test

#SAVE_CHECKPOINT
def save_checkpoint_train(state, filename):
    # print("Saving Checkpoint Train")
    torch.save(state,filename)

def save_checkpoint_test(state, filename):
    # print("Saving Checkpoint Test")
    torch.save(state,filename)

#LOAD_CHECKPOINT
def load_checkpoint_train(checkpoint,model,optimize):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimize.load_state_dict(checkpoint["optimizer"])
    model.eval()
    # 
def load_checkpoint_test(checkpoint,model):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

#CHECK ACCURACY
def iou(outputs,labels):
    sanity = 1e-7
    intersection = torch.sum(torch.logical_and(outputs,labels),dim=(0,1,2,3,4))
    union = torch.sum(torch.logical_or(outputs,labels),dim=(0,1,2,3,4))
    iou = (intersection + sanity)/(union + sanity)
    return (iou)


def check_accuracy(loader,model,device,loss_func):
    model.eval()
    iou_net = 0
    dice_score = 0
    i = 0
    loss_f = 0
    with torch.no_grad():
        for x,y in loader:
            i+=1
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            loss = loss_func(pred,y).item()
            loss_f += loss
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float().detach()
            iou_net += iou(pred,y).item()
            dice_c =  ((2*torch.sum(pred*y))/(torch.sum(pred+y) + 1e-7))
            dice_score += dice_c.item()
            
    # print("Length of Loader: ")
    # print(len(loader))
    # print("Intersection over Union: ")
    # print(iou_net/(len(loader)))
    # print("Dice Score: ")
    # print(dice_score/(len(loader)))
    # print("Loss: ")
    # print(loss_f/(len(loader)))

    return (iou_net/(len(loader)),dice_score/(len(loader)))
    # model.train()

def save_prediction (loader,model,path,device,img_path):

    model.eval()
    i = 0
    loop = tqdm(loader)

    dir_path = img_path[0:img_path.rfind("/")]
    all_ims = list(sorted(os.listdir(dir_path)))

    for batch_idx, (x,y) in enumerate(loop):
        
        x = x.float().to(device)
        n_c = x.shape[2]
        y = y.float().to(device)
        with torch.no_grad():
            x = torch.sigmoid(model(x)).detach()
        
        x = x.squeeze(0)
        x = x.squeeze(0)
        x = (x>0.5).float()
        x = x*255.0
        y = y.squeeze(0)
        y = y.squeeze(0)
        y = y*255.0
        try:
            name = all_ims[i]
        except:
            name = i+".tif"
        change_dims_one(path + "/" + name,x,img_path)
        i+=1
        del x
        del y
        
    print("Saved Predicted Masks")

    model.train()


def save_prediction_test (loader,model,path,device,img_path,truth:bool):

    model.eval()
    i = 0

    loop = tqdm(loader)
    #dir_path = img_path[0:img_path.rfind("/")]
    dir_path = os.path.dirname(img_path)

    all_ims = list(sorted(os.listdir(dir_path)))

    for batch_idx, x in enumerate(loop):

        loop.set_description(f"Image {batch_idx+1}/{len(loop)}: Model Inference")
        # print(all_ims[i])
        x = x.float().to(device)
        n_c = x.shape[2]
        with torch.no_grad():
            x = torch.sigmoid(model(x)).detach()
        
        x = x.squeeze(0)
        x = x.squeeze(0)
        x = (x>0.5).float()
        x = x*255.0

        # imsave(path + "/predicted_mask/" +  str(i) + ".tif",x.detach().cpu().numpy())
        loop.set_description(f"Image {i}/{len(loop)}: Saving Image")
        try:
            name = all_ims[i]
        except:
            name = i+".tif" # TODO : dont not assume .tif it can be .tiff

        if truth:
            change_dims_one_act(path + "/" +  name,x,img_path)
        else:
            change_dims_one(path + "/" +  name,x,img_path)
        i+=1
        del x

        
    print("Saved Predicted Masks")

    model.train()

def get_start(x,max_val):

    start = math.ceil((max_val-x)/2)-1
    if (start!=-1):
        return start
    else:
        return start+1

def change_dims_one_act(path1,img,img_path):

    mask_predicted = img.cpu().detach().numpy()

    # img = Image.open(img_path)
    # org_image = []
    # for i in range (0,img.n_frames):
    #     res = img.seek(i)
    #     org_image.append(np.array(img,dtype=np.float32))
    # org_image = np.array(org_image,dtype=np.float32)

    # d = org_image.shape[0]
    # w = org_image.shape[1]
    # h = org_image.shape[2]

    # mask_array = np.zeros([d,w,h])

    # max_dim = max(max(h,w),d)

    # for i in range (0,d):
    #     temp = mask_predicted[i]
    #     x = cv2.resize(temp, (max_dim, max_dim),interpolation = cv2.INTER_CUBIC)
    #     # x = mask_predicted.resize((608,608),resample =Image.NEAREST)
    #     x_im = (np.array(x))[get_start(w,max_dim):get_start(w,max_dim)+w,get_start(h,max_dim):get_start(h,max_dim)+h]
    #     # print(x_im.shape)
    #     mask_array[i,:,:] = x_im
    
    # mask_array = np.where(mask_array > 0, 1, 0)
    # mask_array = mask_array*255
    mask_array = np.array(mask_predicted,dtype=np.float32)
    imsave(path1,mask_array)
    del mask_array

def change_dims_one(path1,img,img_path):

    mask_predicted = img.cpu().detach().numpy()

    img = Image.open(img_path)
    org_image = []
    for i in range (0,img.n_frames):
        res = img.seek(i)
        org_image.append(np.array(img,dtype=np.float32))
    org_image = np.array(org_image,dtype=np.float32)

    d = org_image.shape[0]
    w = org_image.shape[1]
    h = org_image.shape[2]

    mask_array = np.zeros([d,w,h])

    max_dim = max(max(h,w),d)

    for i in range (0,d):
        temp = mask_predicted[i]
        x = cv2.resize(temp, (max_dim, max_dim),interpolation = cv2.INTER_CUBIC)
        # x = mask_predicted.resize((608,608),resample =Image.NEAREST)
        x_im = (np.array(x))[get_start(w,max_dim):get_start(w,max_dim)+w,get_start(h,max_dim):get_start(h,max_dim)+h]
        # print(x_im.shape)
        mask_array[i,:,:] = x_im
    
    # mask_array = np.where(mask_array > 0, 1, 0)
    # mask_array = mask_array*255
    mask_array = np.array(mask_array,dtype=np.float32)
    imsave(path1,mask_array)
    del mask_array
    del x_im
    del x
    del temp
    del org_image
