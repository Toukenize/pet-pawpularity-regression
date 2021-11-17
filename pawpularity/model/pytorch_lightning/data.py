import os
from pathlib import Path
from typing import List, Optional

import albumentations as a
import cv2
import pandas as pd
import torch
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, dataset

from ...config.constants import IMAGENET_MEAN, IMAGENET_STD, IMG_DIM


class PawData(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            img_folder: Path,
            transforms: Compose,
            img_path_col: str = 'Id',
            label_col: Optional[str] = None,
            meta_cols: Optional[List[str]] = None,
            norm: bool = True):

        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.transforms = transforms
        self.meta_cols = meta_cols
        self.norm = norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Read & transform image
        img_path = self.img_folder / (row[self.img_path_col] + '.jpg')
        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=image)
        image = augmented['image']

        data = dict(image=image)

        # Read & store meta data
        if self.meta_cols is not None:
            data['meta'] = torch.tensor(
                self.df.iloc[idx][self.meta_cols],
                dtype=torch.float32)

        # Read & store label
        if self.label_col is not None:

            label = torch.tensor([row[self.label_col]], dtype=torch.float32)

            if self.norm:
                label /= 100

            data['label'] = label

        return data


def get_train_transforms(img_dim: int) -> Compose:

    trans = a.Compose([
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=120, p=0.8),
        a.RandomBrightnessContrast(p=0.5),
        a.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return trans


def get_val_transforms(img_dim: int) -> Compose:

    trans = a.Compose([
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
        a.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return trans


def get_dataset(
        df: pd.DataFrame,
        img_folder: Path,
        img_dim: int = IMG_DIM,
        is_train: bool = True,
        meta_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
) -> Dataset:

    if is_train:
        trans = get_train_transforms(img_dim)
    else:
        trans = get_val_transforms(img_dim)

    dataset = PawData(
        df=df,
        img_folder=img_folder,
        transforms=trans,
        img_path_col='Id',
        label_col=label_col,
        meta_cols=meta_cols
    )

    return dataset
