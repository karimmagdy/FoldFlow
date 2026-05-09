"""Generate visualization figures for the paper.

Produces:
  - fig4_energy_gates.pdf: Energy gate activation distribution across heads
  - fig5_cross_domain.pdf: Cross-domain component importance comparison
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAVE_DIR = Path("figures")


def plot_cross_domain(save_dir: Path = SAVE_DIR):
    """Bar chart comparing component importance across CIFAR-10 and WikiText-2."""
    components = ["Energy/\nLangevin\n(C1)", "Dynamic\nTopology\n(C2)",
                  "Coopera-\ntivity (C3)", "Chaperone\n(C4)",
                  "Env.\nSensit. (C5)", "Energy\nGate"]

    # CIFAR-10: delta accuracy (pp) when adding component to baseline+C1
    cifar_delta = [0.2, 1.3, 0.5, 0.4, 0.4, 0.0]  # no energy gate for vision

    # WikiText-2: delta PPL when removing component from full model
    # Positive = removing hurts (component helps)
    wt2_delta = [
        0.0,    # removing Langevin: no effect (converged)
        0.0,    # no topology in LM
        0.0,    # no cooperativity in LM
        1.9,    # removing chaperone: 525.2 -> 527.1
        0.0,    # no env sensitivity in LM
        60.9,   # removing energy gate (converged)
    ]

    x = np.arange(len(components))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # CIFAR-10 panel
    colors_c = ["#2196F3" if v > 0 else "#ccc" for v in cifar_delta]
    bars1 = ax1.bar(x, cifar_delta, width=0.6, color=colors_c, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Accuracy Gain (pp)", fontsize=11)
    ax1.set_title("CIFAR-10 (Vision)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=8.5)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylim(-0.5, 2.0)
    for bar, v in zip(bars1, cifar_delta):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                     f"+{v}", ha="center", va="bottom", fontsize=9)

    # WikiText-2 panel
    colors_w = ["#FF5722" if v > 1 else ("#4CAF50" if v < 0 else "#ccc") for v in wt2_delta]
    bars2 = ax2.bar(x, wt2_delta, width=0.6, color=colors_w, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("PPL Increase When Removed", fontsize=11)
    ax2.set_title("WikiText-2 (Language)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(components, fontsize=8.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    for bar, v in zip(bars2, wt2_delta):
        if abs(v) > 0.1:
            label = f"+{v:.1f}" if v > 0 else f"{v:.1f}"
            ax2.text(bar.get_x() + bar.get_width()/2.,
                     max(bar.get_height(), 0) + 1,
                     label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(save_dir / f"fig4_cross_domain.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig4_cross_domain to {save_dir}")


def plot_step_sweep(save_dir: Path = SAVE_DIR):
    """Plot CIFAR-10 accuracy vs Langevin depth."""
    steps = [0, 2, 4, 8]
    accs = []
    for s in steps:
        path = Path(f"results/cifar10_step_sweep/steps_{s}/cifar10_steps_{s}.json")
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            accs.append(d["best_accuracy"])
        else:
            accs.append(None)

    if any(a is None for a in accs):
        print("Step sweep data incomplete, skipping fig5")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(steps, accs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Langevin Steps $T$", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Effect of Dynamics Depth on CIFAR-10", fontsize=13)
    ax.set_xticks(steps)
    ax.set_ylim(81.0, 83.0)
    ax.grid(True, alpha=0.3)

    for s, a in zip(steps, accs):
        ax.annotate(f"{a:.1f}%", (s, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(save_dir / f"fig5_step_sweep.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig5_step_sweep to {save_dir}")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plot_cross_domain(SAVE_DIR)
    plot_step_sweep(SAVE_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
