from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from datasets.CFD.CFDdata import CFD
from datasets.CRACK500.CRACKdata import C5D
import torch
import os


def get_data_loaders(split_seed, img_factor=1, dataset="CFD"):
    # Define standard variables
    current_path = os.path.abspath(os.getcwd())
    image = "/datasets/CFD/cfd_image/"
    gt = "/datasets/CFD/seg_gt/"
    ratio = [71, 47]
    batch_size = 10
    batch_size_t = 47

    if dataset == "CRACK500":
        image = "/datasets/CRACK500/crack_image/"
        gt = "/datasets/CRACK500/seg_gt/"
        ratio = [2021, 1347]

    tf_compos, tf_compos_gt = get_transforms(img_factor)
    if dataset == "CFD":
        dataset = CFD(current_path + image, tf_compos, current_path + gt, tf_compos_gt)
    else:
        dataset = C5D(current_path + image, tf_compos, current_path + gt, tf_compos_gt)

    # Manual seed added such that same split is kept,
    # even though a new split is made with different sizes
    print(f"dataset len: {len(dataset)}")
    train_data, test_data = random_split(dataset, ratio,
                                         generator=torch.Generator().manual_seed(split_seed))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size_t)

    print(
        f"dataset: {len(dataset)}, objects_train: {len(train_data)}, BS_train: {batch_size}, dataloader len: {len(train_loader)}")
    print(f"objects_test: {len(test_data)}, BS_test: {batch_size_t}, dataloader len: {len(test_loader)}")

    return dataset, train_loader, test_loader


def get_transforms(img_factor):
    resize_image = tuple(int(img_factor * x) for x in (320, 480))
    crop_image = resize_image[0]

    shared_transforms = [
        transforms.RandomCrop(crop_image),
        transforms.Pad(200, padding_mode='reflect'),
        transforms.RandomRotation((0, 360)),
        transforms.CenterCrop(crop_image),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    # Bilinear interpolation since value can be [-255, +255]
    tf_compos = transforms.Compose([
        transforms.Resize(resize_image, interpolation=InterpolationMode.BILINEAR),
        *shared_transforms,
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor()
    ])
    # NN interpolation since value can only be [0 or 1],
    # Bilinear, should be tested at some point, however.
    tf_compos_gt = transforms.Compose([
        transforms.Resize(resize_image, interpolation=InterpolationMode.NEAREST),
        *shared_transforms,
        transforms.ToTensor()
    ])
    return tf_compos, tf_compos_gt
