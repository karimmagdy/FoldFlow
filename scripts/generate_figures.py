"""Generate paper figures from ablation and training results.

Produces:
  1. Ablation bar chart (Fig 1)
  2. Training curves (Fig 2)
  3. WikiText-2 comparison table (Fig 3)

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --results-dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = {
    "baseline": "#8e8e8e",
    "+energy": "#4c72b0",
    "+energy+topology": "#55a868",
    "+energy+cooperativity": "#c44e52",
    "+energy+chaperone": "#8172b2",
    "+energy+env_sensitivity": "#ccb974",
    "full_model": "#dd8452",
}


def plot_ablation(results_path: Path, save_dir: Path):
    """Bar chart of ablation results with error bars."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    names = list(results.keys())
    means = [results[n]["mean_accuracy"] for n in names]
    stds = [results[n]["std_accuracy"] for n in names]

    # Display names
    display_names = [
        "Baseline\n(encoder only)",
        "+Energy\n(C1)",
        "+Topology\n(C1+C2)",
        "+Cooperative\n(C1+C3)",
        "+Chaperone\n(C1+C4)",
        "+Environment\n(C1+C5)",
        "Full Model\n(all)",
    ]
    # Truncate if fewer
    display_names = display_names[:len(names)]

    colors = [COLORS.get(n, "#999999") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Annotate values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.15,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("FoldFlow Ablation Study — CIFAR-10")

    # Add baseline reference line
    ax.axhline(y=means[0], color="#8e8e8e", linestyle="--", alpha=0.5, linewidth=1)

    # Zoom y-axis to make differences visible
    y_lo = min(means) - max(stds) - 1.5
    y_hi = max(means) + max(stds) + 1.5
    ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    fig.savefig(save_dir / "fig1_ablation.pdf", bbox_inches="tight")
    fig.savefig(save_dir / "fig1_ablation.png", bbox_inches="tight")
    print(f"Saved ablation figure to {save_dir / 'fig1_ablation.pdf'}")
    plt.close()


def plot_training_curves(results_path: Path, save_dir: Path):
    """Training loss and accuracy curves."""
    with open(results_path) as f:
        data = json.load(f)

    log = data.get("log", [])
    if not log:
        print("No training log found, skipping training curves")
        return

    epochs = [entry["epoch"] for entry in log]
    train_acc = [entry["train_acc"] for entry in log]
    val_acc = [entry["val_acc"] for entry in log]
    train_loss = [entry["train_loss"] for entry in log]

    # EMA accuracy: keep only meaningful entries (> 50% rules out placeholders)
    ema_epochs = [entry["epoch"] for entry in log if entry.get("ema_acc", 0) > 50]
    ema_acc = [entry["ema_acc"] for entry in log if entry.get("ema_acc", 0) > 50]

    # Loss: filter out placeholder zeros from merged text log
    loss_epochs = [entry["epoch"] for entry in log if entry["train_loss"] > 0]
    loss_vals = [entry["train_loss"] for entry in log if entry["train_loss"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy
    ax1.plot(epochs, train_acc, label="Train", alpha=0.7)
    ax1.plot(epochs, val_acc, label="Val", alpha=0.9)
    if ema_epochs:
        ax1.plot(ema_epochs, ema_acc, "s--", label="Val (EMA)", alpha=0.9, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("CIFAR-10 Training Curves")
    ax1.legend()

    # Loss (only epochs with real loss values)
    ax2.plot(loss_epochs, loss_vals, label="Train Loss", color="tab:red", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(save_dir / "fig2_training_curves.pdf", bbox_inches="tight")
    fig.savefig(save_dir / "fig2_training_curves.png", bbox_inches="tight")
    print(f"Saved training curves to {save_dir / 'fig2_training_curves.pdf'}")
    plt.close()


def plot_wikitext2(results_path: Path, save_dir: Path):
    """WikiText-2 LM ablation bar chart."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Desired order for ablation display
    ordered_keys = [
        "transformer++",
        "foldflow-no-energy-gate",
        "foldflow-no-chaperone",
        "foldflow-no-langevin",
        "foldflow",
    ]
    # Fall back to whatever keys exist
    names = [k for k in ordered_keys if k in results]
    if not names:
        names = list(results.keys())

    ppls = [results[n]["mean_ppl"] for n in names]
    stds = [results[n]["std_ppl"] for n in names]
    params = [results[n]["num_parameters"] for n in names]

    display_map = {
        "transformer++": "Transformer++",
        "foldflow-no-energy-gate": "FoldFlow\nw/o energy gate",
        "foldflow-no-chaperone": "FoldFlow\nw/o chaperone",
        "foldflow-no-langevin": "FoldFlow\nw/o Langevin",
        "foldflow": "FoldFlow\n(full)",
    }
    display_names = [display_map.get(n, n) for n in names]

    color_map = {
        "transformer++": "#4c72b0",
        "foldflow-no-energy-gate": "#c44e52",
        "foldflow-no-chaperone": "#8172b2",
        "foldflow-no-langevin": "#ccb974",
        "foldflow": "#dd8452",
    }
    colors = [color_map.get(n, "#999999") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), ppls, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
                  width=0.6)

    for bar, ppl, std in zip(bars, ppls, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 3,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("WikiText-2 Language Modeling — Component Ablation", pad=10)

    # Zoom y-axis
    y_lo = min(ppls) - max(stds) - 20
    y_hi = max(ppls) + max(stds) + 20
    ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    fig.savefig(save_dir / "fig3_wikitext2.pdf", bbox_inches="tight")
    fig.savefig(save_dir / "fig3_wikitext2.png", bbox_inches="tight")
    print(f"Saved WikiText-2 figure to {save_dir / 'fig3_wikitext2.pdf'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--save-dir", default="./figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ablation
    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        plot_ablation(ablation_path, save_dir)
    else:
        print(f"No ablation results at {ablation_path}")

    # Training curves
    training_path = results_dir / "cifar10_results.json"
    if not training_path.exists():
        training_path = Path("checkpoints") / "cifar10_results.json"
    if training_path.exists():
        plot_training_curves(training_path, save_dir)
    else:
        print(f"No training results found")

    # WikiText-2 (prefer ablation results if available)
    wikitext_path = results_dir / "wikitext2_ablation_direct" / "wikitext2_ablation_direct.json"
    if not wikitext_path.exists():
        wikitext_path = results_dir / "wikitext2_results.json"
    if wikitext_path.exists():
        plot_wikitext2(wikitext_path, save_dir)
    else:
        print(f"No WikiText-2 results found")


if __name__ == "__main__":
    main()
