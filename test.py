#increase dimension of training mask at 1
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.functional as F
import os
import threshold
from tqdm import tqdm
import torch.optim as optimizer
from unet import *
from dataset_test import *
from utils import *

def main():
    path = os.getcwd()
    data = input("Enter name of directory containing Data: ")
    cellname = input("Enter name of cell: ")

    pred_mask = os.makedirs(os.path.join(path, "output", cellname, "predicted_mask"), exist_ok=True)
    pred_path = os.path.join(path, "output", cellname, "predicted_mask")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.
    test_loader = get_loaders_test(path,data,cellname)
    model = UNet(in_channels=1,out_channels=1)
    model.to(device)

    checkpt = input("Enter name of the model tar file: ")
    load_checkpoint_test(torch.load(checkpt, map_location=device), model)
    one_img = (list(sorted(os.listdir(os.path.join(path,data,cellname)))))[0]
    img_path = os.path.join(path,data,cellname,one_img)
    
    print(f"Loaded {len(test_loader)} test images from {os.path.join(path, data, cellname)}.")
    print(f"Loaded checkpoint: {checkpt} on {device} device.")
    
    save_prediction_test(test_loader,model,pred_path,device,img_path)
    threshold.run(pred_path) 

if __name__ == "__main__":
    main()
