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

    data = input("Enter parent directory containing data images: ")
    label = input("Enter parent directory containing ground truth masks: ")
    cellname = input("Enter name of cell directory containing images in both parent directory: ")
    part = input("Partition Dataset? (Y/N)")
    
    if (part=="Y"):
        train_loader,test_loader = get_loaders(path,data,label,cellname,part)
    else:
        train_loader = get_loaders(path,data,label,cellname,part)

    num_classes = 2
    num_epochs = int(input("Enter Number of Epochs: "))
    transfer = input("Do you want to train further on trained data?(Y/N) :")
    if (transfer == "Y"):
        load_model = True
    else:
        load_model = False
    scaler = torch.cuda.amp.GradScaler()
    dice_test = []
    iou_test = []
    loss_test = []

    dice_train = []
    iou_train = []
    loss_train = []

    if load_model:
        load_checkpoint_train(torch.load(str(input("Trained Model Name: ")), map_location=device),model,optimize)

    for epoch in range (num_epochs):
        train_one_epoch(train_loader,model,optimize,loss_fn,scaler,device)
        try:
            checkpoint = {"state_dict": model.state_dict(), "optimizer" : optimize.state_dict()}
            save_checkpoint_train(checkpoint)
            print("Saved")
            checkpoint_test = {"state_dict": model.state_dict()}
            save_checkpoint_test(checkpoint_test)
            print("Saved")
        except:
            print("Error")
        finally:
            print("Epoch No. :" + str(epoch))
            if (part=="Y"):
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

    root = os.getcwd()
    one_img = (list(sorted(os.listdir(os.path.join(root,data,cellname)))))[0]
    img_path = os.path.join(path,data,cellname,one_img)
    
    # output_dir = os.path.join(root,"output")
    # if (not os.path.isdir(output_dir)):
    #     os.mkdir(output_dir)
    # cell_dir = os.path.join(output_dir,cellname)
    # if (not os.path.isdir(cell_dir)):
    #     os.mkdir(cell_dir)
    # pred_mask = os.path.join(cell_dir,"predicted_mask")
    # if (not os.path.isdir(pred_mask)):
    #     os.mkdir(pred_mask)
    
    pred_mask = os.path.join(root, "output", cellname, "predicted_mask")
    os.makedirs(pred_mask, exist_ok=True)
    
    if (part == "Y"):
        print("Test Trends: ")
        print("Dice: ")
        print(dice_test)
        print("IOU: ")
        print(iou_test)
        print("Loss: ")
        print(loss_test)

    print("Train Trends: ")
    print("Dice: ")
    print(dice_train)
    print("IOU: ")
    print(iou_train)
    print("Loss: ")
    print(loss_train)

    print("Saving Test Predictions")
    save_prediction (test_loader,model,pred_mask,device,img_path) 
    print("Saved")        


if __name__ == "__main__":
    main()
