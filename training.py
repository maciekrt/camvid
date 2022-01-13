
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import argparse

import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader

from models import UNet
from metrics import CamvidAccuracy


data_paths = {
    "train": {
        "images": Path("/notebooks/data/CamVid/train/"),
        "masks": Path("/notebooks/data/CamVid/train_labels/")
    },
    "test": {
        "images": Path("/notebooks/data/CamVid/test/"),
        "masks": Path("/notebooks/data/CamVid/test_labels/")
    },
    "val": {
        "images": Path("/notebooks/data/CamVid/val/"),
        "masks": Path("/notebooks/data/CamVid/val_labels/")
    },
}


class LossModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, masks_logits, masks_gt):
        return self.cross_entropy_loss(masks_logits, masks_gt) # + dice_loss(masks_logits, masks_gt)


class UNetModel(pl.LightningModule):
    
    def __init__(self, size, learning_rate=3e-4, steps_scheduler=100):
        super().__init__()
        self.size = size
        self.learning_rate = learning_rate
        self.save_hyperparameters() 
        self.model = resnet.resnet34(pretrained=True)
        self.unet = UNet(
            encoder=self.model,
            out_channels=32,
            size=self.size
        )
        self.steps_scheduler = steps_scheduler
        self.camvid_accuracy = CamvidAccuracy()
        self.loss = LossModule()
        
    def forward(self, x):
        mask_probs = self.unet(x)
        return mask_probs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             pct_start=0.9,
#             max_lr=3e-3,
#             steps_per_epoch=self.steps_scheduler,
#             epochs=20
#         )
        return [optimizer] #, [scheduler]
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, masks_gt, _ = batch
        masks_logits = self.unet(imgs)
        loss = self.loss(masks_logits, masks_gt)
        self.log("train_loss", loss)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, masks_gt, _ = batch
        masks_logits = self.unet(imgs)
        loss = self.loss(masks_logits, masks_gt)
        preds = masks_logits.argmax(dim=1)
        self.camvid_accuracy(preds, masks_gt)
        self.log("val_loss", loss)
        self.log("val_acc", self.camvid_accuracy)
        return {"loss": loss, "acc": self.camvid_accuracy}
    
    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, masks_gt, _ = batch
        masks_logits = self.unet(imgs)
        loss = self.loss(masks_logits, masks_gt)
        preds = masks_logits.argmax(dim=1)
        self.camvid_accuracy(preds, masks_gt)
        self.log("test_acc", self.camvid_accuracy)
        self.log("test_loss", loss)
        return {"loss": loss, "acc": self.camvid_accuracy}
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="display a square of a given number")
    args = parser.parse_args()
    
    size = (720, 960)
    
    # Training dataset and dataloader
    ds_train = CamvidDataset(data_paths["train"]["images"], data_paths["train"]["masks"], transform_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    
    # Validation dataset and dataloader
    ds_val = CamvidDataset(data_paths["val"]["images"], data_paths["val"]["masks"], transform_val)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    
    # Test dataset and dataloader (comes with images)
    
    # Loading the model
    unet_pl_model = UNetModel(size=size, learning_rate=1e-3, steps_scheduler=len(dl_train))
    
    # Model checkpoint callback (we are keeping track of accuracy)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='unet-{epoch:02d}-{val_acc:.2f}'
    )
    # Early stopping of patience 5
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.0,
        patience=5,
        mode="max",
    )
    
    # Training for 30 epochs
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(unet_pl_model, dl_train, dl_val)
