"""CIFAR-100 data loading with modern augmentation (CutMix, MixUp, label smoothing).

Mirrors the CIFAR-10 loader but with 100 classes.
"""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar100_transforms(train: bool = True):
    """Standard CIFAR-100 transforms with AutoAugment for training."""
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),  # same policy works for CIFAR-100
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-100 train/test loaders using FULL 50K training set."""
    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=get_cifar100_transforms(train=True),
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=get_cifar100_transforms(train=False),
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader
