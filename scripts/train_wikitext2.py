"""Train FoldFlow vs Transformer++ on WikiText-2 with IDENTICAL hyperparameters.

Fair comparison: same d_model, n_layers, seq_len, batch_size, epochs, lr for all.

Usage:
    python scripts/train_wikitext2.py
    python scripts/train_wikitext2.py --epochs 20 --seeds 3
    python scripts/train_wikitext2.py --quick  # 3 epochs, 1 seed
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldflow.models.lm import LMConfig, FoldFlowLM, TransformerPPLM
from foldflow.data.wikitext2 import load_wikitext2
from foldflow.utils.training import (
    AverageMeter, cosine_lr_schedule, count_parameters, Timer,
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
    parser = argparse.ArgumentParser(description="WikiText-2: FoldFlow vs T++")

    # Shared model config (IDENTICAL for both)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training (IDENTICAL for both)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--energy-steps", type=int, default=3,
                        help="FoldFlow Langevin refinement steps at inference")
    parser.add_argument("--disable-energy-gate", action="store_true",
                        help="Disable FoldFlow energy gating for ablations")
    parser.add_argument("--disable-chaperone", action="store_true",
                        help="Disable FoldFlow chaperone correction for ablations")
    parser.add_argument("--disable-langevin", action="store_true",
                        help="Disable FoldFlow Langevin refinement for ablations")

    # General
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=42,
                        help="First seed value (subsequent seeds are +1, +2, ...)")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--results-name", type=str, default="wikitext2_results.json",
                        help="Filename for JSON metrics under save-dir")
    parser.add_argument("--quick", action="store_true", help="Quick: 3 epochs, 1 seed")
    parser.add_argument("--models", nargs="+", default=["transformer++", "foldflow"],
                        help="Which models to train")

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    loss_meter = AverageMeter()
    total_tokens = 0
    timer = Timer()
    sync_device(device)
    timer.start()

    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        _, loss = model(input_ids, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_meter.update(loss.item(), input_ids.size(0))
        total_tokens += (labels != -100).sum().item()

    sync_device(device)
    epoch_time = timer.elapsed()
    return {
        "loss": loss_meter.avg,
        "epoch_time_s": epoch_time,
        "tokens_per_s": format_throughput(total_tokens, epoch_time),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    total_tokens = 0
    timer = Timer()
    sync_device(device)
    timer.start()

    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        _, loss = model(input_ids, labels)
        # Count non-padding tokens for accurate PPL
        n_tokens = (labels != -100).sum().item()
        loss_meter.update(loss.item(), n_tokens)
        total_tokens += n_tokens

    sync_device(device)
    eval_time = timer.elapsed()
    ppl = math.exp(min(loss_meter.avg, 20))  # cap to avoid inf
    return {
        "loss": loss_meter.avg,
        "ppl": ppl,
        "eval_time_s": eval_time,
        "tokens_per_s": format_throughput(total_tokens, eval_time),
    }


def make_foldflow_config(args, vocab_size):
    return LMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        energy_steps=args.energy_steps,
        use_energy_gate=not args.disable_energy_gate,
        use_chaperone=not args.disable_chaperone,
        use_langevin=not args.disable_langevin,
    )


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 3
        args.seeds = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, vocab_size = load_wikitext2(
        seq_len=args.seq_len, batch_size=args.batch_size,
    )

    # Shared config
    shared_config = LMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        energy_steps=args.energy_steps,
    )

    model_factories = {
        "transformer++": lambda: TransformerPPLM(shared_config),
        "foldflow": lambda: FoldFlowLM(make_foldflow_config(args, vocab_size)),
        "foldflow-no-energy-gate": lambda: FoldFlowLM(
            make_foldflow_config(
                argparse.Namespace(**{**vars(args), "disable_energy_gate": True}), vocab_size
            )
        ),
        "foldflow-no-chaperone": lambda: FoldFlowLM(
            make_foldflow_config(
                argparse.Namespace(**{**vars(args), "disable_chaperone": True}), vocab_size
            )
        ),
        "foldflow-no-langevin": lambda: FoldFlowLM(
            make_foldflow_config(
                argparse.Namespace(**{**vars(args), "disable_langevin": True}), vocab_size
            )
        ),
    }

    all_results = {}
    timer = Timer()
    timer.start()

    for model_name in args.models:
        if model_name not in model_factories:
            print(f"Unknown model: {model_name}, skipping")
            continue

        seed_results = []
        for seed_idx in range(args.seeds):
            seed = args.seed_start + seed_idx
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Seed: {seed}")

            model = model_factories[model_name]().to(device)
            n_params = count_parameters(model)
            print(f"Parameters: {n_params:,}")

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )

            best_ppl = float("inf")
            epoch_log = []
            for epoch in range(args.epochs):
                lr = cosine_lr_schedule(
                    optimizer, epoch, args.epochs,
                    warmup_epochs=args.warmup_epochs, base_lr=args.lr,
                )
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, device, args.grad_clip,
                )
                val_metrics = evaluate(model, val_loader, device)
                best_ppl = min(best_ppl, val_metrics["ppl"])

                epoch_log.append({
                    "epoch": epoch + 1,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_epoch_time_s": train_metrics["epoch_time_s"],
                    "train_tokens_per_s": train_metrics["tokens_per_s"],
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["ppl"],
                    "val_eval_time_s": val_metrics["eval_time_s"],
                    "val_tokens_per_s": val_metrics["tokens_per_s"],
                    "best_ppl": best_ppl,
                })

                if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
                    print(
                        f"  Epoch {epoch+1:3d}/{args.epochs} | "
                        f"LR {lr:.6f} | "
                        f"Train Loss {train_metrics['loss']:.4f} | "
                        f"Val Loss {val_metrics['loss']:.4f} | "
                        f"Val PPL {val_metrics['ppl']:.1f} | "
                        f"Best PPL {best_ppl:.1f}"
                    )

            seed_results.append({
                "seed": seed,
                "best_ppl": best_ppl,
                "final_val_loss": val_metrics["loss"],
                "num_parameters": n_params,
                "train_epoch_time_s": float(np.mean([entry["train_epoch_time_s"] for entry in epoch_log])),
                "train_tokens_per_s": float(np.mean([entry["train_tokens_per_s"] for entry in epoch_log])),
                "val_eval_time_s": float(np.mean([entry["val_eval_time_s"] for entry in epoch_log])),
                "val_tokens_per_s": float(np.mean([entry["val_tokens_per_s"] for entry in epoch_log])),
                "log": epoch_log,
            })

        ppls = [r["best_ppl"] for r in seed_results]
        all_results[model_name] = {
            "seeds": seed_results,
            "mean_ppl": float(np.mean(ppls)),
            "std_ppl": float(np.std(ppls)),
            "num_parameters": seed_results[0]["num_parameters"],
            "mean_train_epoch_time_s": float(np.mean([r["train_epoch_time_s"] for r in seed_results])),
            "mean_train_tokens_per_s": float(np.mean([r["train_tokens_per_s"] for r in seed_results])),
            "mean_val_eval_time_s": float(np.mean([r["val_eval_time_s"] for r in seed_results])),
            "mean_val_tokens_per_s": float(np.mean([r["val_tokens_per_s"] for r in seed_results])),
        }
        print(f"\n{model_name}: PPL = {np.mean(ppls):.1f} ± {np.std(ppls):.1f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'PPL':>15} {'Params':>12}")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"{name:<20} {res['mean_ppl']:>8.1f} ± {res['std_ppl']:>4.1f}  {res['num_parameters']:>10,}")
    print(f"{'='*60}")
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"seq_len={args.seq_len}, batch={args.batch_size}, epochs={args.epochs}")
    print(f"Total time: {timer.elapsed():.0f}s")

    # Save
    save_path = Path(args.save_dir) / args.results_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "config": {
                "d_model": args.d_model, "n_layers": args.n_layers,
                "n_heads": args.n_heads, "seq_len": args.seq_len,
                "batch_size": args.batch_size, "epochs": args.epochs,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "energy_steps": args.energy_steps,
                "disable_energy_gate": args.disable_energy_gate,
                "disable_chaperone": args.disable_chaperone,
                "disable_langevin": args.disable_langevin,
            },
            "results": all_results,
            "total_time_s": timer.elapsed(),
        }, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
