"""Train WikiText-2 models to convergence with full ablation suite.

Runs Transformer++ baseline, full FoldFlow, and all ablation variants for
100 epochs (configurable) with 5 seeds, cosine annealing LR + warmup,
early stopping, and bootstrap confidence intervals.

Usage:
    python scripts/train_wikitext2_convergence.py
    python scripts/train_wikitext2_convergence.py --epochs 100 --seeds 5
    python scripts/train_wikitext2_convergence.py --model foldflow
    python scripts/train_wikitext2_convergence.py --model ablation_all
    python scripts/train_wikitext2_convergence.py --quick  # 20 epochs, 2 seeds
"""

import argparse
import json
import logging
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


# ---------------------------------------------------------------------------
# Model group definitions
# ---------------------------------------------------------------------------

MODEL_GROUPS = {
    "baseline": ["transformer++"],
    "foldflow": ["foldflow"],
    "ablation_all": [
        "transformer++",
        "foldflow",
        "foldflow-no-energy-gate",
        "foldflow-no-chaperone",
        "foldflow-no-langevin",
    ],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("convergence")


def setup_logging(save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(save_dir) / "wikitext2_convergence.log"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, rng=None):
    """Compute bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.array(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        m = float(values.mean())
        return m, m, m
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(values.mean()), lo, hi


# ---------------------------------------------------------------------------
# Training / evaluation (mirrors train_wikitext2.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_foldflow_config(args, vocab_size, **overrides):
    """Build LMConfig for a FoldFlow variant, with optional flag overrides."""
    return LMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        energy_steps=args.energy_steps,
        use_energy_gate=overrides.get("use_energy_gate", True),
        use_chaperone=overrides.get("use_chaperone", True),
        use_langevin=overrides.get("use_langevin", True),
    )


def build_model_factories(args, vocab_size):
    shared_config = LMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        energy_steps=args.energy_steps,
    )
    return {
        "transformer++": lambda: TransformerPPLM(shared_config),
        "foldflow": lambda: FoldFlowLM(make_foldflow_config(args, vocab_size)),
        "foldflow-no-energy-gate": lambda: FoldFlowLM(
            make_foldflow_config(args, vocab_size, use_energy_gate=False)
        ),
        "foldflow-no-chaperone": lambda: FoldFlowLM(
            make_foldflow_config(args, vocab_size, use_chaperone=False)
        ),
        "foldflow-no-langevin": lambda: FoldFlowLM(
            make_foldflow_config(args, vocab_size, use_langevin=False)
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiText-2 convergence training with full ablation suite"
    )

    # Model config (identical for all variants)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training config
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Warmup epochs for cosine annealing schedule")
    parser.add_argument("--energy-steps", type=int, default=3,
                        help="FoldFlow Langevin refinement steps at inference")

    # Convergence / early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val PPL improvement)")

    # Experiment control
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42,
                        help="First seed value (subsequent seeds are +1, +2, ...)")
    parser.add_argument("--model", type=str, default="ablation_all",
                        choices=["foldflow", "baseline", "ablation_all"],
                        help="Which model group to train")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--results-name", type=str,
                        default="wikitext2_convergence.json",
                        help="Filename for JSON results under save-dir")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 epochs, 2 seeds, patience=5")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_single_run(model_name, model, args, train_loader, val_loader, device, seed):
    """Train one model to convergence with early stopping.

    Returns a dict with per-seed results including full epoch log.
    """
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    best_ppl = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    epoch_log = []

    for epoch in range(args.epochs):
        # Cosine annealing with warmup
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs, base_lr=args.lr,
        )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device)

        improved = val_metrics["ppl"] < best_ppl
        if improved:
            best_ppl = val_metrics["ppl"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

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

        # Log every epoch for training curve analysis
        log.info(
            f"  [{model_name}/s{seed}] Epoch {epoch+1:3d}/{args.epochs} | "
            f"LR {lr:.6f} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Val PPL {val_metrics['ppl']:.1f} | "
            f"Best PPL {best_ppl:.1f} (ep{best_epoch})"
        )

        # Early stopping
        if epochs_no_improve >= args.patience:
            log.info(
                f"  [{model_name}/s{seed}] Early stopping at epoch {epoch+1} "
                f"(no improvement for {args.patience} epochs)"
            )
            break

    return {
        "seed": seed,
        "best_ppl": best_ppl,
        "best_epoch": best_epoch,
        "final_epoch": epoch + 1,
        "final_val_loss": val_metrics["loss"],
        "num_parameters": n_params,
        "early_stopped": epochs_no_improve >= args.patience,
        "mean_train_epoch_time_s": float(np.mean(
            [e["train_epoch_time_s"] for e in epoch_log]
        )),
        "mean_train_tokens_per_s": float(np.mean(
            [e["train_tokens_per_s"] for e in epoch_log]
        )),
        "mean_val_eval_time_s": float(np.mean(
            [e["val_eval_time_s"] for e in epoch_log]
        )),
        "mean_val_tokens_per_s": float(np.mean(
            [e["val_tokens_per_s"] for e in epoch_log]
        )),
        "log": epoch_log,
    }


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 20
        args.seeds = 2
        args.patience = 5

    setup_logging(args.save_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(f"Device: {device}")
    log.info(f"Epochs: {args.epochs}, Seeds: {args.seeds}, Patience: {args.patience}")
    log.info(f"Model group: {args.model}")

    # Load data
    train_loader, val_loader, vocab_size = load_wikitext2(
        seq_len=args.seq_len, batch_size=args.batch_size,
    )

    # Determine which models to run
    models_to_run = MODEL_GROUPS[args.model]
    model_factories = build_model_factories(args, vocab_size)

    all_results = {}
    global_timer = Timer()
    global_timer.start()

    total_runs = len(models_to_run) * args.seeds
    run_idx = 0

    for model_name in models_to_run:
        if model_name not in model_factories:
            log.info(f"Unknown model: {model_name}, skipping")
            continue

        seed_results = []

        for seed_offset in range(args.seeds):
            seed = args.seed_start + seed_offset
            run_idx += 1
            set_seed(seed)

            log.info(f"\n{'='*70}")
            log.info(f"[{run_idx}/{total_runs}] Model: {model_name} | Seed: {seed}")
            log.info(f"{'='*70}")

            model = model_factories[model_name]().to(device)
            result = train_single_run(
                model_name, model, args, train_loader, val_loader, device, seed,
            )
            seed_results.append(result)

            log.info(
                f"  -> Best PPL {result['best_ppl']:.1f} at epoch {result['best_epoch']} "
                f"| Final epoch {result['final_epoch']} "
                f"| Params {result['num_parameters']:,}"
            )

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate per-model stats with bootstrap CIs
        ppls = [r["best_ppl"] for r in seed_results]
        mean_ppl, ci_lo, ci_hi = bootstrap_ci(ppls)

        all_results[model_name] = {
            "seeds": seed_results,
            "mean_ppl": mean_ppl,
            "std_ppl": float(np.std(ppls)),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "num_parameters": seed_results[0]["num_parameters"],
            "mean_best_epoch": float(np.mean([r["best_epoch"] for r in seed_results])),
            "mean_final_epoch": float(np.mean([r["final_epoch"] for r in seed_results])),
            "early_stopped_count": sum(1 for r in seed_results if r["early_stopped"]),
            "mean_train_epoch_time_s": float(np.mean(
                [r["mean_train_epoch_time_s"] for r in seed_results]
            )),
            "mean_train_tokens_per_s": float(np.mean(
                [r["mean_train_tokens_per_s"] for r in seed_results]
            )),
            "mean_val_eval_time_s": float(np.mean(
                [r["mean_val_eval_time_s"] for r in seed_results]
            )),
            "mean_val_tokens_per_s": float(np.mean(
                [r["mean_val_tokens_per_s"] for r in seed_results]
            )),
        }
        log.info(
            f"\n{model_name}: PPL = {mean_ppl:.1f} +/- {np.std(ppls):.1f} "
            f"[95% CI: {ci_lo:.1f}, {ci_hi:.1f}]"
        )

        # Incremental save after each model
        _save_results(args, all_results, global_timer.elapsed())

    # Final summary table
    log.info(f"\n{'='*80}")
    log.info(
        f"{'Model':<28} {'PPL':>10} {'95% CI':>18} {'Params':>12} {'BestEp':>8}"
    )
    log.info(f"{'='*80}")
    for name, res in all_results.items():
        log.info(
            f"{name:<28} "
            f"{res['mean_ppl']:>7.1f} +/- {res['std_ppl']:.1f} "
            f"[{res['ci_95_lo']:.1f}, {res['ci_95_hi']:.1f}] "
            f"{res['num_parameters']:>10,} "
            f"{res['mean_best_epoch']:>6.0f}"
        )
    log.info(f"{'='*80}")
    log.info(
        f"Config: d_model={args.d_model}, n_layers={args.n_layers}, "
        f"seq_len={args.seq_len}, batch={args.batch_size}, "
        f"epochs={args.epochs}, patience={args.patience}, seeds={args.seeds}"
    )
    log.info(f"Total time: {global_timer.elapsed():.0f}s")
    log.info(f"Results saved to {Path(args.save_dir) / args.results_name}")


def _save_results(args, all_results, elapsed):
    """Save current results to JSON (called incrementally)."""
    save_path = Path(args.save_dir) / args.results_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "config": {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "patience": args.patience,
                "energy_steps": args.energy_steps,
                "seeds": args.seeds,
                "seed_start": args.seed_start,
                "model_group": args.model,
            },
            "results": all_results,
            "total_time_s": elapsed,
        }, f, indent=2)


if __name__ == "__main__":
    main()
