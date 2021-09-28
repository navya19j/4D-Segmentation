#increase dimension of training mask at 1
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch.optim as optimizer
from unet import *
from dataset_test import *
from utils import *

def main():
    path = os.getcwd()
    data = input("Enter name of directory containing Data: ")
    cellname = input("Enter name of cell: ")
    pred_mask = os.path.join(path,"predicted_mask")
    if (not os.path.isdir(pred_mask)):
        os.mkdir(pred_mask)

    learning_rate = 1e-4
    # Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.
    test_loader = get_loaders_test(path,data,cellname)
    model = UNet(in_channels=1,out_channels=1)
    model.to("cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    load_checkpoint_test(torch.load("checkpoint_1.pth.tar"),model)
    #check_accuracy(test_loader,model,"cuda",loss_fn)
    one_img = (list(sorted(os.listdir(os.path.join(path,data,cellname)))))[0]
    # print(len(test_loader))
    img_path = os.path.join(path,data,cellname,one_img)
    save_prediction_test(test_loader,model,path,"cuda",img_path) 

if __name__ == "__main__":
    main()