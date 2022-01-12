
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet

import albumentations as A
from albumentations.pytorch import ToTensorV2


path_classes = Path("/notebooks/data/CamVid/class_dict.csv")
df = pd.read_csv(path_classes)
labels = df.values[:,1:].astype('int16').transpose()

def mask_into_classes(mask):
    comparison = np.expand_dims(mask,axis=-1) == labels
    indicators = np.all(comparison, axis=2)
    classes = np.argmax(indicators, axis=2)
    return classes


class CamvidDataset(Dataset):
    
    def __init__(self, images_path, masks_path, transform, augment=None, test=False):
        """
        Dataset for Camvid problem. 
        
        Args:
            images_path: path to images 
            masks_path: path to masks
            transform: transformation and augmentations to be applied
                before inference
            test: is it a test dataset
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.images_files = [path for path in self.images_path.glob("*.png")]
        self.masks_files = [masks_path / f"{path.stem}_L.png" for path in self.images_files]
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.images_files[idx]))
        mask_rgb = np.asarray(Image.open(self.masks_files[idx]))
        mask = mask_into_classes(mask_rgb)
        if self.augment is None:
            transformed = self.transform(image=image, mask=mask)             
            return transformed["image"], transformed["mask"], image
        else:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            transformed = self.transform(image=image, mask=mask)
            return transformed["image"], transformed["mask"], image
