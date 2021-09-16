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
    optimize = optim.Adam(model.parameters(), lr = learning_rate,weight_decay = 1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    load_checkpoint(torch.load("checkpoint_1.pth.tar"),model,optimize)
    #check_accuracy(test_loader,model,"cuda",loss_fn)
    save_prediction_test (test_loader,model,path,"cuda") 

if __name__ == "__main__":
    main()