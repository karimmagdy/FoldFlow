"""Train FoldFlow on CIFAR-100 — baseline + full model comparison.

Usage:
    python scripts/train_cifar100.py --epochs 50 --seeds 3
    python scripts/train_cifar100.py --quick  # 10 epochs, 1 seed
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldflow.models.classifier import FoldFlowClassifier
from foldflow.data.cifar100 import get_cifar100_loaders
from foldflow.data.cifar10 import cutmix_data, mixup_data, mixup_criterion
from foldflow.utils.training import (
    AverageMeter, accuracy, cosine_lr_schedule, EMA,
    count_parameters, Timer, format_throughput, sync_device,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Train FoldFlow on CIFAR-100")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-particles", type=int, default=64)
    p.add_argument("--particle-dim", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--encoder", type=str, default="weak", choices=["weak", "strong"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="./results/cifar100")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, args, ema):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    num_samples = 0
    timer = Timer()
    sync_device(device)
    timer.start()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
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
        if use_mix:
            loss = mixup_criterion(criterion, logits, targets, targets_b, lam)
        else:
            loss = criterion(logits, targets)
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
    return {"loss": loss_meter.avg, "acc": acc_meter.avg,
            "epoch_time_s": timer.elapsed(),
            "samples_per_s": format_throughput(num_samples, timer.elapsed())}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)["logits"]
        loss_meter.update(criterion(logits, targets).item(), images.size(0))
        acc_meter.update(accuracy(logits, targets), images.size(0))
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


MODEL_CONFIGS = {
    "baseline": dict(use_energy=False, use_topology=False, use_cooperativity=False,
                     use_chaperone=False, use_env_sensitivity=False),
    "full_model": dict(use_energy=True, use_topology=True, use_cooperativity=True,
                       use_chaperone=True, use_env_sensitivity=True),
}


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 10
        args.seeds = 1

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f"CIFAR-100: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    all_results = {}
    global_timer = Timer()
    global_timer.start()

    for config_name, flags in MODEL_CONFIGS.items():
        seed_results = []
        for seed_idx in range(args.seeds):
            seed = 42 + seed_idx
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Config: {config_name} | Seed: {seed}")

            model = FoldFlowClassifier(
                num_classes=100,
                hidden_dim=args.hidden_dim,
                num_particles=args.num_particles,
                particle_dim=args.particle_dim,
                num_steps=args.num_steps,
                channels=3,
                dropout=args.dropout,
                encoder_type=args.encoder,
                **flags,
            ).to(device)

            n_params = count_parameters(model)
            print(f"Parameters: {n_params:,}")

            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            ema = EMA(model, decay=args.ema_decay)

            best_acc = 0.0
            for epoch in range(args.epochs):
                lr = cosine_lr_schedule(optimizer, epoch, args.epochs,
                                        warmup_epochs=args.warmup_epochs,
                                        base_lr=args.lr)
                train_one_epoch(model, train_loader, optimizer, criterion,
                                device, args, ema)
                val = evaluate(model, test_loader, criterion, device)
                best_acc = max(best_acc, val["acc"])

                if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                          f"Val {val['acc']:.1f}% | Best {best_acc:.1f}%")

                # EMA eval at end
                if epoch == args.epochs - 1:
                    ema_val = evaluate(ema.shadow, test_loader, criterion, device)
                    best_acc = max(best_acc, ema_val["acc"])
                    print(f"  EMA: {ema_val['acc']:.1f}%")

            seed_results.append({
                "seed": seed,
                "best_accuracy": best_acc,
                "num_parameters": n_params,
            })
            print(f"  Final best: {best_acc:.2f}%")

        accs = [s["best_accuracy"] for s in seed_results]
        all_results[config_name] = {
            "flags": flags,
            "seeds": seed_results,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "num_parameters": seed_results[0]["num_parameters"],
        }
        print(f"\n{config_name}: {np.mean(accs):.1f} ± {np.std(accs):.1f}%")

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "cifar100_results.json"
    with open(results_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {global_timer.elapsed():.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("CIFAR-100 Summary:")
    for name, r in all_results.items():
        print(f"  {name}: {r['mean_accuracy']:.1f} ± {r['std_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
