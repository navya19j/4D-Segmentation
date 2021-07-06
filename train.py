#increase dimension of training mask at 1
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
import torchio as tio


#try deleting loss after each epoch taking it out of loop and summing

#Hyperparameters
torch.set_printoptions(profile="full")

def train_one_epoch(loader,model,optimizer,loss_func,scaler,device):

    loop = tqdm(loader)

    for batch_idx , (x,y) in enumerate(loop):

        x = x.float().to(device)
        y = y.float().to(device)

        with torch.cuda.amp.autocast():
            x = model(x)
            loss = loss_func(x,y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())
        
        del x 
        del y
        del loss


def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if (device=="cuda"):
        torch.cuda.empty_cache()
        
    model = UNet(in_channels=1,out_channels=1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = 1e-4
    optimize = optim.Adam(model.parameters(), lr = learning_rate,weight_decay = 1e-5)
    path = os.getcwd()
    train_loader = get_loaders(path)[0]
    test_loader = get_loaders(path)[1]

    num_classes = 2
    num_epochs = 15
    load_model = False
    scaler = torch.cuda.amp.GradScaler()
    dice_test = []
    iou_test = []
    loss_test = []

    dice_train = []
    iou_train = []
    loss_train = []

    if load_model:
        load_checkpoint(torch.load("try.pth.tar"),model)
        # print("Accuracy on Test: ")
        # x,y,z = check_accuracy(test_loader,model,device,loss_fn)
        # dice_test.append(x)
        # iou_test.append(y)
        # loss_test.append(z)
        # print("Accuracy on Train: ")
        # a,b,c = check_accuracy(train_loader,model,device,loss_fn)
        # dice_train.append(a)
        # iou_train.append(b)
        # loss_train.append(c)
            
        # save_prediction (test_loader,model,path,device) 

    for epoch in range (num_epochs):

        train_one_epoch(train_loader,model,optimize,loss_fn,scaler,device)
        try:
            checkpoint = {"state_dict": model.state_dict(), "optimizer" : optimize.state_dict()}
            save_checkpoint(checkpoint)
        except:
            print("Error")
        finally:
            print("Accuracy on Test: ")
            x,y,z = check_accuracy(test_loader,model,device,loss_fn)
            dice_test.append(x)
            iou_test.append(y)
            loss_test.append(z)

            print("Accuracy on Train: ")
            a,b,c = check_accuracy(train_loader,model,device,loss_fn)
            dice_train.append(a)
            iou_train.append(b)
            loss_train.append(c)

            
    save_prediction (test_loader,model,path,device) 
    print("Test Trends: ")
    print("Dice: " + dice_test)
    print("IOU: " + iou_test)
    print("Loss: " + loss_test)

    print("Train Trends: ")
    print("Dice: " + dice_train)
    print("IOU: " + iou_train)
    print("Loss: " + loss_train)

    print("Saving")
    save_prediction (test_loader,model,path,device) 
    print("Saved")        


if __name__ == "__main__":
    main()

