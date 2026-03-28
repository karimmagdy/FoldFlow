"""CIFAR-10 data loading with modern augmentation (CutMix, MixUp, label smoothing).

Uses the FULL 50K training set (old version only used 25K).
"""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar10_transforms(train: bool = True):
    """Standard CIFAR-10 transforms with AutoAugment for training."""
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train/test loaders using FULL 50K training set."""
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_cifar10_transforms(train=True),
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_cifar10_transforms(train=False),
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


# ---------------------------------------------------------------------------
# CutMix / MixUp
# ---------------------------------------------------------------------------

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation.

    Returns mixed images, label pairs, and lambda for loss computation.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    B = x.size(0)
    index = torch.randperm(B, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    return x_mixed, y[index], lam


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply MixUp augmentation."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[index]
    return x_mixed, y[index], lam


def mixup_criterion(
    criterion, logits: torch.Tensor,
    y_a: torch.Tensor, y_b: torch.Tensor, lam: float,
) -> torch.Tensor:
    """Compute mixed loss for CutMix/MixUp."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
