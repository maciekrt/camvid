
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F

import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader


def add_mask_from_rle(full_mask, rle, label):
    sy, sx = full_mask.shape
    for (offset, length) in rle:
        x = (offset-1) % sx
        y = (offset-1) // sx
        full_mask[y,x:x+length] = label
        
def encode_rles(masks_rle, height, width):
    mask = np.zeros((height, width), dtype=np.int32)
    for j, rle in enumerate(masks_rle):
        elems = rle.split()
        elems = [(int(elems[2*i]), int(elems[2*i+1])) for i in range(len(elems)//2)]
        add_mask_from_rle(mask, elems, 1)
    return mask

# Important paths
# data_path = Path("/notebooks/data/sartorius")
# train_path = data_path / "train"
# output_path = Path("/notebooks/data/sartorius/masks")
# output_path.mkdir(exist_ok=True)


# df_descr = pd.read_csv(data_path / "train.csv")
# data = []
# for img_path in tqdm(train_path.glob("*.png")):
#     cell_desc = df_descr[df_descr["id"] == img_path.stem]["annotation"]
#     full_mask = encode_rles(cell_desc, 520, 704)
#     img = np.asarray(Image.open(str(img_path)).convert("RGB"))
#     data.append({
#         "img_path": img_path,
#         "image": img,
#         "mask": full_mask
#     })

# image_paths = list(train_path.glob("*.png"))

# class ImageDataset(Dataset):
    
#     def __init__(self, data, transform=None):
#         self.data = image_paths
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img = data[idx]["image"].copy()
#         mask = torch.from_numpy(data[idx]["mask"].copy()).float()
#         if self.transform is None:
#             return img, mask
#         else:
#             return self.transform(img), mask
        
        
# def dataloaders():
#     dataset = ImageDataset(
#         image_paths, 
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#     )
#     train_set, val_set = torch.utils.data.random_split(dataset, [510,96])
#     train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
#     valid_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)
#     return train_dataloader, valid_dataloader

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

path_classes = Path("/notebooks/data/CamVid/class_dict.csv")
df = pd.read_csv(path_classes)
labels = df.values[:,1:].astype('int16').transpose()
print("Masks codes loaded.")
print(labels.shape)


def mask_into_classes(mask):
    comparison = np.expand_dims(mask,axis=-1) == labels
    indicators = np.all(comparison, axis=2)
    classes = np.argmax(indicators, axis=2)
    return classes

    
def get_data(phase="train"):
    
    if phase not in ["train", "test", "val"]:
        raise ValueError(f"Data is not available for such a phase [{phase}]")
    
    print(f"Loading {phase} data..")
    images_path = data_paths[phase]["images"]
    masks_path = data_paths[phase]["masks"]
    
    images_pairs = [(path, masks_path / f"{path.stem}_L.png" ) for path in images_path.glob("*.png")]
    data = []
    for (path_image, path_mask) in tqdm(images_pairs):
        image = Image.open(path_image)
        mask = Image.open(path_mask)
        image_data = np.asarray(image)
        mask_data = np.asarray(mask)
        classes = mask_into_classes(mask_data)
        data.append({"path": path_image, "image": image_data, "classes": classes})
    return data
    

class CamVidDataset(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"].copy()
        classes = torch.from_numpy(self.data[idx]["classes"].copy())
        if self.transform is None:
            return img, classes
        else:
            return self.transform(img), classes


def dataloaders():
    loaders = {}
    datasets = {}
    for label in ["train", "val", "test"]:
        data = get_data(label)
        dataset = CamVidDataset(
            data, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        datasets[label] = dataset
    loaders["train"] = DataLoader(datasets["train"], batch_size=8, shuffle=True, drop_last=False, num_workers=4)
    loaders["val"] = DataLoader(datasets["val"], batch_size=8, shuffle=False, drop_last=False, num_workers=4)
    loaders["test"] = DataLoader(datasets["test"], batch_size=8, shuffle=False, drop_last=False, num_workers=4)
    return loaders

if __name__=="__main__":
    loaders = dataloaders()
    val_loaders = loaders["val"]
    print(next(iter(val_loaders)))