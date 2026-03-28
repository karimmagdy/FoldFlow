"""Training utilities: metrics, logging, cosine schedule, EMA."""

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy (%)."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item() * 100


def cosine_lr_schedule(optimizer, epoch: int, total_epochs: int,
                       warmup_epochs: int = 5, base_lr: float = 0.001, min_lr: float = 1e-6):
    """Update learning rate with linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        # Ensure shadow is on the same device as the model
        device = next(model.parameters()).device
        self.shadow.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(m_param, alpha=1 - self.decay)
        # Copy buffers (e.g. BatchNorm running stats) directly from source model
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self):
        return self.shadow.state_dict()


def save_checkpoint(model: nn.Module, optimizer, epoch: int, acc: float,
                    path: Path, ema: EMA = None):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": acc,
    }
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time


def sync_device(device: torch.device):
    """Synchronize the active accelerator for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def format_throughput(num_items: int, duration_s: float) -> float:
    """Return processed items per second."""
    if duration_s <= 0:
        return 0.0
    return num_items / duration_s
