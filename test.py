import os

import torch

import threshold
from dataset_test import *
from unet import *
from utils import *

# increase dimension of training mask at 1

def main(config):

    # load test config
    ROOT = os.getcwd()
    TEST_DATA = config["test_data"]
    CELLNAME = config["cellname"]
    TRUTH = config["truth"]
    NUM_LAYERS = config["num_layers"]
    CHECKPOINT_FILE = config["checkpoint"]

    # data
    pred_path = os.path.join(ROOT, "output", CELLNAME, "predicted_mask")
    pred_mask = os.makedirs(
        os.path.join(ROOT, "output", CELLNAME, "predicted_mask"), exist_ok=True
    )

    # Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.
    test_loader = get_loaders_test(ROOT, TEST_DATA, CELLNAME, TRUTH)
    print(
        f"Loaded {len(test_loader)} test images from {os.path.join(ROOT, TEST_DATA, CELLNAME)}."
    )

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(NUM_LAYERS, in_channels=1, out_channels=1)
    model.to(device)
    load_checkpoint_test(torch.load(CHECKPOINT_FILE, map_location=device), model)
    print(f"Loaded checkpoint: {CHECKPOINT_FILE} on {device} device.")

    # output directory
    one_img = (list(sorted(os.listdir(os.path.join(ROOT, TEST_DATA, CELLNAME)))))[0]
    img_path = os.path.join(ROOT, TEST_DATA, CELLNAME, one_img)

    # run testing
    save_prediction_test(test_loader, model, pred_path, device, img_path, TRUTH)
    threshold.run(pred_path)


if __name__ == "__main__":
    main()
