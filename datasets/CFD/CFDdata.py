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
        self.img_fnames = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = glob.glob(os.path.join(mask_dir, "*.png"))
        self.mask_transform = mask_transform

        self.seed = torch.manual_seed(15)

    def __getitem__(self, i):
        fpath = self.img_fnames[i]
        img = Image.open(fpath).convert("RGB")
        if self.img_transform is not None:
            self.seed
            img = self.img_transform(img)

        mpath = self.mask_fnames[i]
        mask = Image.open(mpath).convert("1")
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_fnames)


# if __name__ == "__main__":
#     img_dir = pathlib.Path("cfd_image")
#     img_transform = None

#     mask_dir = pathlib.Path("seg_gt")
#     mask_transform = None
#     img, mask = CFD(img_dir, img_transform, mask_dir, mask_transform)[0]    # get the first img, mask tuplet
#     print(f"image shape: {img.size}\nmask shape: {mask.size}")