"""Run ablation experiments: 7 configs × 3 seeds = 21 runs.

Configs:
  1. Baseline: encoder-only (all components OFF)
  2. +Energy (C1)
  3. +Energy +Topology (C1+C2)
  4. +Energy +Cooperativity (C1+C3)
  5. +Energy +Chaperone (C1+C4)
  6. +Energy +EnvSensitivity (C1+C5)
  7. Full model (all ON)

Each run: 100 epochs (shorter than full training for feasibility).
Results saved as JSON for paper figures.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --epochs 100 --seeds 3
    python scripts/run_ablation.py --quick  # 20 epochs, 1 seed
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldflow.models.classifier import FoldFlowClassifier
from foldflow.data.cifar10 import get_cifar10_loaders
from foldflow.utils.training import (
    AverageMeter, accuracy, cosine_lr_schedule, count_parameters, Timer,
)


ABLATION_CONFIGS = {
    "baseline": dict(
        use_energy=False, use_topology=False,
        use_cooperativity=False, use_chaperone=False, use_env_sensitivity=False,
    ),
    "+energy": dict(
        use_energy=True, use_topology=False,
        use_cooperativity=False, use_chaperone=False, use_env_sensitivity=False,
    ),
    "+energy+topology": dict(
        use_energy=True, use_topology=True,
        use_cooperativity=False, use_chaperone=False, use_env_sensitivity=False,
    ),
    "+energy+cooperativity": dict(
        use_energy=True, use_topology=False,
        use_cooperativity=True, use_chaperone=False, use_env_sensitivity=False,
    ),
    "+energy+chaperone": dict(
        use_energy=True, use_topology=False,
        use_cooperativity=False, use_chaperone=True, use_env_sensitivity=False,
    ),
    "+energy+env_sensitivity": dict(
        use_energy=True, use_topology=False,
        use_cooperativity=False, use_chaperone=False, use_env_sensitivity=True,
    ),
    "full_model": dict(
        use_energy=True, use_topology=True,
        use_cooperativity=True, use_chaperone=True, use_env_sensitivity=True,
    ),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="FoldFlow Ablation Study")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--particle-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--encoder", type=str, default="weak", choices=["weak", "strong"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 20 epochs, 1 seed")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Run specific configs only, e.g. --configs baseline full_model")
    return parser.parse_args()


log = logging.getLogger("ablation")


def setup_logging(save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(save_dir) / "ablation_full_log.txt"
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(sh)


def train_and_evaluate(model, train_loader, test_loader, args, device, label=""):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        cosine_lr_schedule(optimizer, epoch, args.epochs, warmup_epochs=5, base_lr=args.lr)

        # Train
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            result = model(images)
            loss = criterion(result["logits"], targets)
            if "aux_logits" in result:
                for aux in result["aux_logits"]:
                    loss += 0.1 * criterion(aux, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate
        model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)["logits"]
                acc_meter.update(accuracy(logits, targets), images.size(0))
        best_acc = max(best_acc, acc_meter.avg)

        # Per-epoch progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"    {label} Epoch {epoch+1}/{args.epochs} | Acc {acc_meter.avg:.2f}% | Best {best_acc:.2f}%")

    return best_acc


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 20
        args.seeds = 1

    setup_logging(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Epochs: {args.epochs}, Seeds: {args.seeds}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir,
    )

    configs_to_run = args.configs or list(ABLATION_CONFIGS.keys())

    # Load existing results for resume support
    save_path = Path(args.save_dir) / "ablation_results.json"
    all_results = {}
    if save_path.exists() and args.configs:
        with open(save_path) as f:
            existing = json.load(f)
        all_results = existing.get("results", {})
        log.info(f"Loaded {len(all_results)} existing configs from {save_path}")

    total = len(configs_to_run) * args.seeds
    done = 0
    timer = Timer()
    timer.start()

    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            log.info(f"Unknown config: {config_name}, skipping")
            continue

        flags = ABLATION_CONFIGS[config_name]
        seed_results = []

        for seed in range(args.seeds):
            done += 1
            set_seed(seed + 42)
            log.info(f"\n[{done}/{total}] Config: {config_name} | Seed: {seed + 42}")

            model = FoldFlowClassifier(
                num_classes=10,
                hidden_dim=args.hidden_dim,
                num_particles=args.num_particles,
                particle_dim=args.particle_dim,
                num_steps=args.num_steps,
                channels=3,
                encoder_type=args.encoder,
                **flags,
            ).to(device)

            num_params = count_parameters(model)
            best_acc = train_and_evaluate(model, train_loader, test_loader, args, device, label=f"{config_name}/s{seed+42}")
            seed_results.append({
                "seed": seed + 42,
                "best_accuracy": best_acc,
                "num_parameters": num_params,
            })
            log.info(f"  -> Accuracy: {best_acc:.2f}% | Params: {num_params:,}")

        accs = [r["best_accuracy"] for r in seed_results]
        all_results[config_name] = {
            "flags": flags,
            "seeds": seed_results,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "num_parameters": seed_results[0]["num_parameters"],
        }
        log.info(f"  Config {config_name}: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")

        # Incremental save after each config
        save_path = Path(args.save_dir) / "ablation_results.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Summary table
    log.info(f"\n{'='*70}")
    log.info(f"{'Config':<30} {'Accuracy':>12} {'Params':>12}")
    log.info(f"{'='*70}")
    for name, res in all_results.items():
        log.info(f"{name:<30} {res['mean_accuracy']:>8.2f} +/- {res['std_accuracy']:.2f}  {res['num_parameters']:>10,}")
    log.info(f"{'='*70}")
    log.info(f"Total time: {timer.elapsed():.0f}s")

    # Save
    save_path = Path(args.save_dir) / "ablation_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
        }, f, indent=2)
    log.info(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
