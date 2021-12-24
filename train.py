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

        loop.set_description(f"Loading Batch {batch_idx+1}/{len(loop)}")
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

def main(train_inputs):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if (device=="cuda"):
        torch.cuda.empty_cache()

    # num = int(input("Number of Layers: "))
    num,data,label,cellname,part,truth,bat_size,num_epochs,transfer,trained_model = int(train_inputs[0]),train_inputs[1],train_inputs[2],train_inputs[3],train_inputs[4],train_inputs[5],int(train_inputs[6]),int(train_inputs[7]),train_inputs[8],train_inputs[9]
    # data = input("Enter parent directory containing data images: ")
    # label = input("Enter parent directory containing ground truth masks: ")
    # cellname = input("Enter name of cell directory containing images in both parent directory: ")
    # part = input("Partition Dataset? (Y/N)")
    # truth = input("Do you want to test on actual size? (Y/N)")
    # bat_size = int(input("Batch Size: "))
    # num_epochs = int(input("Enter Number of Epochs: "))
    # transfer = input("Do you want to train further on trained data?(Y/N) :")
    # trained_model = str(input("Trained Model Name: "))

    model = UNet(layers=num,in_channels=1,out_channels=1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = 1e-4
    optimize = optim.Adam(model.parameters(), lr = learning_rate,weight_decay = 1e-5)
    path = os.getcwd()

    if (part=="Y"):
        train_loader,test_loader = get_loaders(path,data,label,cellname,part,truth,bat_size)
    else:
        train_loader = get_loaders(path,data,label,cellname,part,truth,bat_size)
        test_loader = train_loader

    num_classes = 2

    if (transfer == "Y"):
        load_model = True
    else:
        load_model = False
    scaler = torch.cuda.amp.GradScaler()
    dice_test = 0
    iou_test = 0

    dice_train = 0
    iou_train = 0

    if load_model:
        load_checkpoint_train(torch.load(trained_model, map_location=device),model,optimize)

    epochs = [i for i in range (num_epochs)]
    loop = tqdm(epochs)

    for epoch in loop:

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(train_loader,model,optimize,loss_fn,scaler,device)
 
        x,y = check_accuracy(test_loader,model,device,loss_fn)
        a,b = check_accuracy(train_loader,model,device,loss_fn)

        if (y>iou_test or x>dice_test):
            checkpoint = {"state_dict": model.state_dict(), "optimizer" : optimize.state_dict()}
            save_checkpoint_train(checkpoint)
            checkpoint_test = {"state_dict": model.state_dict()}
            save_checkpoint_test(checkpoint_test)
            loop.set_postfix({'IOU Test':y,'Dice Test':x})
            iou_test = y
            dice_test = x
            iou_train = b
            dice_train = a


        loop.set_postfix({'IOU Train': b,'Dice Train':a})

    root = os.getcwd()
    one_img = (list(sorted(os.listdir(os.path.join(root,data,cellname)))))[0]
    img_path = os.path.join(path,data,cellname,one_img)

    pred_mask = os.path.join(root, "output", cellname, "predicted_mask")
    os.makedirs(pred_mask, exist_ok=True)
    
    if (part == "Y"):
        print("Test Trends: ")
        print("Dice: ")
        print(dice_test)
        print("IOU: ")
        print(iou_test)

    print("Train Trends: ")
    print("Dice: ")
    print(dice_train)
    print("IOU: ")
    print(iou_train)
 


if __name__ == "__main__":
    main()

