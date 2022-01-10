import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from unet import *
from utils import *

# increase dimension of training mask at 1
# try deleting loss after each epoch taking it out of loop and summing

# Hyperparameters
torch.set_printoptions(profile="full")


def train_one_epoch(loader, model, optimizer, loss_func, scaler, device):

    model.train()
    loop = tqdm(loader)
    for batch_idx, (x, y) in enumerate(loop):

        loop.set_description(f"Loading Batch {batch_idx+1}/{len(loop)}")
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.cuda.amp.autocast():
            x = model(x)
            loss = loss_func(x, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

        del x
        del y
        del loss


def main(config):

    # load training config
    # data
    ROOT = os.getcwd()
    TRAIN_DATA = config["train_data"]
    LABEL_DATA = config["label_data"]
    CELLNAME = config["cellname"]

    # model
    CHECKPOINT_FILE = config["checkpoint"]
    TRAIN_TEST_SPLIT = config["train_test_split"]
    DOWNSAMPLE = config["downsample"] # atm this is NOT_DOWNSAMPLE

    # hyperparameters
    NUM_LAYERS = config["num_layers"]
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # load model
    model = UNet(layers=NUM_LAYERS, in_channels=1, out_channels=1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimize = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler()

    if CHECKPOINT_FILE:
        load_checkpoint_train(
            torch.load(CHECKPOINT_FILE, map_location=device), model, optimize
        )

    # load dataset
    if TRAIN_TEST_SPLIT:
        train_loader, test_loader = get_loaders(
            ROOT, TRAIN_DATA, LABEL_DATA, CELLNAME, TRAIN_TEST_SPLIT, DOWNSAMPLE, BATCH_SIZE
        )
    else:
        train_loader = get_loaders(
            ROOT, TRAIN_DATA, LABEL_DATA, CELLNAME, TRAIN_TEST_SPLIT, DOWNSAMPLE, BATCH_SIZE
        )
        test_loader = train_loader

    # num_classes = 2

    dice_test = 0
    iou_test = 0

    dice_train = 0
    iou_train = 0

    loop = tqdm(range(NUM_EPOCHS))
    for epoch in loop:

        loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_one_epoch(train_loader, model, optimize, loss_fn, scaler, device)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimize.state_dict(),
        }
        checkpoint_test = {"state_dict": model.state_dict()}

        x, y = check_accuracy(test_loader, model, device, loss_fn)
        a, b = check_accuracy(train_loader, model, device, loss_fn)
        print(x)
        print(y)

        loop.set_postfix({"IOU Test": y, "Dice Test": x})

        if y >= iou_test or x >= dice_test:
            save_checkpoint_train(checkpoint, "checkpoint_best_train.pth.tar")
            save_checkpoint_test(checkpoint_test, "checkpoint_best_test.pth.tar")
            iou_test = y
            dice_test = x
            iou_train = b
            dice_train = a
            print("Saved")

        loop.set_postfix({"IOU Train": b, "Dice Train": a})

    # ROOT = os.getcwd()
    # one_img = (list(sorted(os.listdir(os.path.join(ROOT,TRAIN_DATA,CELLNAME)))))[0]
    # img_path = os.path.join(ROOT,TRAIN_DATA,CELLNAME,one_img)

    pred_mask = os.path.join(ROOT, "output", CELLNAME, "predicted_mask")
    os.makedirs(pred_mask, exist_ok=True)

    if TRAIN_TEST_SPLIT:
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
