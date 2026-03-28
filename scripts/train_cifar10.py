"""Train FoldFlow on CIFAR-10.

Full training: 200 epochs, 50K samples, CutMix/MixUp, label smoothing 0.1,
cosine schedule with 5-epoch warmup, EMA.

Usage:
    python scripts/train_cifar10.py
    python scripts/train_cifar10.py --encoder strong --epochs 200
    python scripts/train_cifar10.py --quick  # 30 epochs for smoke-testing
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldflow.models.classifier import FoldFlowClassifier
from foldflow.data.cifar10 import (
    get_cifar10_loaders, cutmix_data, mixup_data, mixup_criterion,
)
from foldflow.utils.training import (
    AverageMeter, accuracy, cosine_lr_schedule, EMA,
    save_checkpoint, count_parameters, Timer,
    format_throughput, sync_device,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train FoldFlow on CIFAR-10")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--particle-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--encoder", type=str, default="weak", choices=["weak", "strong"])

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ema-eval-every", type=int, default=10,
                        help="Evaluate EMA model every N epochs (saves ~50%% eval time)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-name", type=str, default="cifar10_results.json",
                        help="Filename for JSON metrics under save-dir")
    parser.add_argument("--checkpoint-name", type=str, default="foldflow_cifar10_best.pt",
                        help="Filename for the best checkpoint under save-dir")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 30 epochs")

    return parser.parse_args()


def train_one_epoch(
    model, loader, optimizer, criterion, device, args, epoch, ema,
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    num_samples = 0
    timer = Timer()
    sync_device(device)
    timer.start()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        # Randomly apply CutMix or MixUp (50/50)
        use_mix = random.random() < 0.5
        if use_mix:
            if random.random() < 0.5:
                images, targets_b, lam = cutmix_data(images, targets, args.cutmix_alpha)
            else:
                images, targets_b, lam = mixup_data(images, targets, args.mixup_alpha)
        else:
            targets_b, lam = targets, 1.0

        result = model(images)
        logits = result["logits"]

        # Main loss
        if use_mix:
            loss = mixup_criterion(criterion, logits, targets, targets_b, lam)
        else:
            loss = criterion(logits, targets)

        # Auxiliary loss from intermediate predictions
        if "aux_logits" in result:
            for aux in result["aux_logits"]:
                if use_mix:
                    loss += args.aux_weight * mixup_criterion(criterion, aux, targets, targets_b, lam)
                else:
                    loss += args.aux_weight * criterion(aux, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(model)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accuracy(logits, targets), images.size(0))
        num_samples += images.size(0)

    sync_device(device)
    epoch_time = timer.elapsed()
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "epoch_time_s": epoch_time,
        "samples_per_s": format_throughput(num_samples, epoch_time),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    num_samples = 0
    timer = Timer()
    sync_device(device)
    timer.start()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        result = model(images)
        logits = result["logits"]
        loss = criterion(logits, targets)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accuracy(logits, targets), images.size(0))
        num_samples += images.size(0)

    sync_device(device)
    eval_time = timer.elapsed()
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "eval_time_s": eval_time,
        "samples_per_s": format_throughput(num_samples, eval_time),
    }


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 30

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = FoldFlowClassifier(
        num_classes=10,
        hidden_dim=args.hidden_dim,
        num_particles=args.num_particles,
        particle_dim=args.particle_dim,
        num_steps=args.num_steps,
        channels=3,
        dropout=args.dropout,
        encoder_type=args.encoder,
    ).to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    ema = EMA(model, decay=args.ema_decay)

    best_acc = 0.0
    start_epoch = 0
    results_log = []

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("accuracy", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best acc: {best_acc:.2f}%)")

    timer = Timer()
    timer.start()

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs, base_lr=args.lr,
        )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args, epoch, ema,
        )

        # Evaluate model
        val_metrics = evaluate(model, test_loader, criterion, device)

        # EMA evaluation (only every N epochs to save time)
        ema_acc = 0.0
        if (epoch + 1) % args.ema_eval_every == 0 or epoch == args.epochs - 1:
            ema_metrics = evaluate(ema.shadow, test_loader, criterion, device)
            ema_acc = ema_metrics["acc"]

        current_best = max(val_metrics["acc"], ema_acc)
        is_best = current_best > best_acc
        best_acc = max(best_acc, current_best)

        log = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_epoch_time_s": train_metrics["epoch_time_s"],
            "train_samples_per_s": train_metrics["samples_per_s"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_eval_time_s": val_metrics["eval_time_s"],
            "val_samples_per_s": val_metrics["samples_per_s"],
            "ema_acc": ema_acc,
            "best_acc": best_acc,
        }
        results_log.append(log)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = timer.elapsed()
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"LR {lr:.6f} | "
                f"Train {train_metrics['acc']:.1f}% | "
                f"Val {val_metrics['acc']:.1f}% | "
                f"EMA {ema_acc:.1f}% | "
                f"Best {best_acc:.1f}% | "
                f"{elapsed:.0f}s"
            )

        # Save best
        if is_best:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                Path(args.save_dir) / args.checkpoint_name, ema,
            )

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete — Best accuracy: {best_acc:.2f}%")
    print(f"Parameters: {num_params:,}")
    print(f"Total time: {timer.elapsed():.0f}s")

    # Save results
    results_path = Path(args.save_dir) / args.results_name
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "args": vars(args),
            "best_accuracy": best_acc,
            "num_parameters": num_params,
            "total_time_s": timer.elapsed(),
            "log": results_log,
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
