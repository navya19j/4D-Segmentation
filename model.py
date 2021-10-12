#!/usr/bin/env python3

from unet import UNet
import torch
from typing import *

class SegmentationModel(object):


    def __init__(self, checkpoint: str = None, mode: str = "train") -> None:
        super().__init__()

        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model(checkpoint=checkpoint)
        # TODO: inference transforms?

    def load_model(self, checkpoint: Optional[str]) -> None:

        self.model = UNet(in_channels=1,out_channels=1)
        self.model.to(self.device)

        if checkpoint:
            checkpoint_state = torch.load(checkpoint, map_location=self.device)

            self.model.load_state_dict(checkpoint_state["state_dict"])
            self.model.eval()
            
            if self.mode == "train":
                # TODO: pass state to optimizer
                self.model.train()


    def pre_process(self, img: torch.Tensor) -> torch.Tensor:
        # img is tensor? maybe not
        
        img_t = torch.Tensor(img).float().to(self.device)
        img_t = img_t.unsqueeze(0) # add batch dim

        return img_t

    def inference(self, img: torch.Tensor) -> torch.Tensor:
        
        img_t = self.pre_process(img)

        output = torch.sigmoid(self.model(img_t)).detach()
        
        output = output.squeeze(0).squeeze(0)
        output = (output > 0.5).float()
        mask = output*255.0 

        return mask


if __name__ == "__main__":

    model = SegmentationModel(checkpoint="checkpoint_train.pth.tar", mode="train")



    # size: 128, 128