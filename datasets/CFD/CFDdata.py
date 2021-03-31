import os
import glob
import pathlib
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CFD(Dataset):
    def __init__(self, img_dir, img_transform, mask_dir, mask_transform):
        self.img_dir = img_dir
        self.img_fnames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.mask_transform = mask_transform

    def __getitem__(self, i: int):
        fpath = self.img_fnames[i]
        img = Image.open(fpath).convert("RGB")
        state = torch.get_rng_state()
        if self.img_transform is not None:
            img = self.img_transform(img)

        mpath = self.mask_fnames[i]
        mask = Image.open(mpath).convert("1")
        torch.set_rng_state(state)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_fnames)