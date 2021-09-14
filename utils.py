from numpy.core.fromnumeric import size
import torch
import torchvision
import os
from tqdm import tqdm
import torch.optim as optimizer
from unet import *
from dataset import *
import numpy as np
from PIL import Image, ImageSequence
import sys
from tifffile import imsave
import torchvision.transforms as TF_v
torch.set_printoptions(profile="full")
from dataset_test import *
# import torchio as tio

def get_loaders(path):
    
    dataset_all = EndosomeDataset(path,None)
    torch.manual_seed(5)

    indices = torch.randperm(len(dataset_all)).tolist()
    if (117 in indices):
        indices.remove(117)
    if (66 in indices):
        indices.remove(66)
    length = len(indices)
    idx = int(-0.2*length)

    dataset_train = torch.utils.data.Subset(dataset_all, indices[:idx])

    dataset_test = torch.utils.data.Subset(dataset_all, indices[idx:])
    print(indices[idx:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=True,pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,num_workers=0, shuffle=False,pin_memory=True)

    return data_loader_train,data_loader_test

def get_loaders_test(path,data,cellname):
    
    dataset_all = Dataset_Test(path,data,cellname)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_all, batch_size=1,num_workers=0, shuffle=False,pin_memory=True)

    return data_loader_test

#SAVE_CHECKPOINT
def save_checkpoint(state, filename = "checkpoint_1.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state,filename)

#LOAD_CHECKPOINT
def load_checkpoint(checkpoint,model):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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
            
    print("Length of Loader: ")
    print(len(loader))
    print("Intersection over Union: ")
    print(iou_net/(len(loader)))
    print("Dice Score: ")
    print(dice_score/(len(loader)))
    print("Loss: ")
    print(loss_f/(len(loader)))

    return (iou_net/(len(loader)),dice_score/(len(loader)),loss_f/(len(loader)))
    model.train()

def save_prediction (loader,model,path,device):

    model.eval()
    i = 0
    for batch_idx, (x,y) in enumerate(loader):
        i+=1
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

        imsave(path + "/predicted_mask/" +  str(batch_idx) +str(i) + ".tif",x.detach().cpu().numpy())
        imsave(path + "/actual_mask/" + str(batch_idx) +str(i) + ".tif",y.detach().cpu().numpy())

        change_dims(path + "/predicted_mask/" +  str(batch_idx) +str(i) + ".tif",path + "/actual_mask/"+ str(batch_idx) +str(i) + ".tif")

    model.train()


def save_prediction_test (loader,model,path,device):

    model.eval()
    i = 0
    for batch_idx, x in enumerate(loader):
        i+=1
        x = x.float().to(device)
        n_c = x.shape[2]

        with torch.no_grad():
            x = torch.sigmoid(model(x)).detach()
        
        x = x.squeeze(0)
        x = x.squeeze(0)
        x = (x>0.5).float()
        x = x*255.0

        # imsave(path + "/predicted_mask/" +  str(i) + ".tif",x.detach().cpu().numpy())
        change_dims_one(path + "/predicted_mask/" +  str(i) + ".tif",x)
        del x

    model.train()

def change_dims_one(path1,img):

        mask_predicted = img.cpu().detach().numpy()
        d = 80
        h_mask,w_mask = 400,608
        mask_array = np.zeros([d,w_mask,h_mask])

        for i in range (0,80):
            temp = mask_predicted[i]
            x = cv2.resize(temp, (608, 608),interpolation = cv2.INTER_CUBIC)
            # x = mask_predicted.resize((608,608),resample =Image.NEAREST)
            x_im = np.array(x)[:,103:503]
            mask_array[i,:,:] = x_im
        
        # mask_array = np.where(mask_array > 0, 1, 0)
        # mask_array = mask_array*255

        imsave(path1,mask_array)
        del mask_array 
        print("done")  

def change_dims(path1,path2):

        mask_predicted = Image.open(path1)
        mask_actual = Image.open(path2)
        d = 80
        h_mask,w_mask = 400,608
        mask_array = np.zeros([d,w_mask,h_mask])
        img_array = np.zeros([d,w_mask,h_mask])

        for i in range (0,80):
            mask_predicted.seek(i)
            x = mask_predicted.resize((608,608),resample =Image.NEAREST)
            x_im = np.array(x)[:,103:503]
            img_array[i,:,:] = x_im

            mask_actual.seek(i)
            y = mask_actual.resize((608,608),resample =Image.NEAREST)
            y_im = np.array(y)[:,103:503]
            mask_array[i,:,:] = y_im
            
        imsave(path1,img_array)
        imsave(path2,mask_array)      
