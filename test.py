#increase dimension of training mask at 1
import os

import torch
import threshold
from dataset_test import *
from unet import *
from utils import *


def main(test_inputs):
    path = os.getcwd()

    # data = input("Enter name of directory containing Data: ")
    # cellname = input("Enter name of cell: ")
    # truth = input("Do you want to test on actual size(Y/N)?")
    # num = int(input("Number of Layers in UNET model : "))
    # checkpt = input("Enter name of the model tar file: ")

    data,cellname,truth,num,checkpt = test_inputs[0],test_inputs[1],test_inputs[2],int(test_inputs[3]),test_inputs[4]

    pred_path = os.path.join(path, "output", cellname, "predicted_mask")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mask = os.makedirs(os.path.join(path, "output", cellname, "predicted_mask"), exist_ok=True)

    # Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.
    test_loader = get_loaders_test(path,data,cellname,truth)
    
    model = UNet(num,in_channels=1,out_channels=1)
    model.to(device)

    
    load_checkpoint_test(torch.load(checkpt, map_location=device), model)
    one_img = (list(sorted(os.listdir(os.path.join(path,data,cellname)))))[0]
    img_path = os.path.join(path,data,cellname,one_img)
    
    print(f"Loaded {len(test_loader)} test images from {os.path.join(path, data, cellname)}.")
    print(f"Loaded checkpoint: {checkpt} on {device} device.")
    
    save_prediction_test(test_loader,model,pred_path,device,img_path,truth)
    threshold.run(pred_path) 

if __name__ == "__main__":
    main()
