
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import Dataset, DataLoader

from models import UNet
from datasets.dataloaders import dataloaders


eps = 0.000001
def compute_ap(predictions, truth):
    iou = compute_iou(predictions, truth)
    n1, n2 = iou.shape
    threshold = 0.5
    precisions = []
    for elem in range(10):
        matches = iou > threshold-eps
        tp = np.sum(matches)
        fp = n1 - tp
        fn = n2 - tp
        precisions.append(tp/(tp+fp+fn))
        threshold += 0.05
    return np.mean(precisions)


def compute_iou(data1, data2):
    n1 = np.max(data1)
    n2 = np.max(data2)
    intersections = np.histogram2d(data1.flatten(), data2.flatten(), bins=(n1+1,n2+1))[0][1:,1:]
    sizes1 = np.histogram(data1.flatten(), bins=n1+1)[0][1:]
    sizes1 = np.expand_dims(sizes1, axis=1)
    sizes2 = np.histogram(data2.flatten(), bins=n2+1)[0][1:]
    sizes2 = np.expand_dims(sizes2, axis=0)
    union = (sizes1 + sizes2 - intersections)
    return intersections/union


def dice_loss(preds, mask, apply_sigmoid=True):
    if apply_sigmoid:
        probs = torch.sigmoid(preds)
    else:
        probs = preds
    intersection = (2*probs*mask).sum(axis=(-2,-1))
    union = (probs**2 + mask**2).sum(axis=(-2,-1))
    return 1.0 - torch.mean(intersection / union)


def acc_camvid(input, target, void_code=30):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


class LossModule(nn.Module):
    
    def __init__(self):
        super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, masks_logits, masks_gt):
        return self.cross_entropy_loss(masks_logits, masks_gt)
#     + dice_loss(masks_logits, masks_gt)
    


class UNetModel(pl.LightningModule):
    
    def __init__(self, model, size, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.size = size
        self.unet = UNet(
            encoder=self.model,
            out_channels=32,
            size=self.size
        )
        self.loss = LossModule()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        mask_probs = self.unet(x)
        return mask_probs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, masks_gt = batch
        masks_logits = self.unet(imgs).squeeze(1)
        loss = self.loss(masks_logits, masks_gt)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, masks_gt = batch
        masks_logits = self.unet(imgs).squeeze(1)
        loss = self.loss(masks_logits, masks_gt)
        acc = acc_camvid(masks_logits, masks_gt)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss
    
    
if __name__=="__main__":
    
    size = (720, 960)
    model = resnet.resnet34(pretrained=True)
    unet_pl_model = UNetModel(model=model, size=size)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='unet-{epoch:02d}-{val_loss:.2f}'
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.005,
        patience=3,
        mode="max",
    )
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
        callbacks=[checkpoint_callback, early_stopping_callback],
#         auto_lr_find=True,
    )
#     train_dataloader, valid_dataloader = dataloaders()
#     trainer.fit(unet_pl_model, train_dataloader, valid_dataloader)
    loaders = dataloaders()
    trainer.fit(unet_pl_model, loaders["train"], loaders["val"])


