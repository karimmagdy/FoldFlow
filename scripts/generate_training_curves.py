"""Generate WikiText-2 training curve figures from JSON results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = "results/wikitext2_ablation_direct/wikitext2_ablation_direct.json"
OUTPUT_DIR = Path("figures")

MODEL_LABELS = {
    "transformer++": "Transformer++",
    "foldflow-no-energy-gate": "FoldFlow w/o energy gate",
    "foldflow-no-chaperone": "FoldFlow w/o chaperone",
    "foldflow-no-langevin": "FoldFlow w/o Langevin",
    "foldflow": "FoldFlow LM (full)",
}

COLORS = {
    "transformer++": "#888888",
    "foldflow-no-energy-gate": "#e07b39",
    "foldflow-no-chaperone": "#2ca02c",
    "foldflow-no-langevin": "#9467bd",
    "foldflow": "#d62728",
}

def load_data():
    with open(RESULTS_FILE) as f:
        return json.load(f)

def plot_ppl_curves(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for model_name in MODEL_LABELS:
        if model_name not in data["results"]:
            continue
        res = data["results"][model_name]
        seeds = res["seeds"]

        # Collect per-epoch PPL across seeds
        n_epochs = len(seeds[0]["log"])
        epochs = np.arange(1, n_epochs + 1)
        ppl_matrix = np.array([[ep["val_ppl"] for ep in s["log"]] for s in seeds])
        mean_ppl = ppl_matrix.mean(axis=0)

        label = MODEL_LABELS[model_name]
        color = COLORS[model_name]
        lw = 2.5 if model_name in ("transformer++", "foldflow") else 1.5
        ls = "-" if model_name in ("transformer++", "foldflow") else "--"

        ax1.plot(epochs, mean_ppl, label=label, color=color, lw=lw, ls=ls)

        if len(seeds) > 1:
            std_ppl = ppl_matrix.std(axis=0)
            ax1.fill_between(epochs, mean_ppl - std_ppl, mean_ppl + std_ppl,
                             alpha=0.15, color=color)

        # Train loss
        loss_matrix = np.array([[ep["train_loss"] for ep in s["log"]] for s in seeds])
        mean_loss = loss_matrix.mean(axis=0)
        ax2.plot(epochs, mean_loss, label=label, color=color, lw=lw, ls=ls)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Perplexity")
    ax1.set_title("WikiText-2 Validation Perplexity")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(1, n_epochs)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("WikiText-2 Training Loss")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(1, n_epochs)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        outpath = OUTPUT_DIR / f"fig_wt2_training_curves.{ext}"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved {outpath}")
    plt.close()

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    plot_ppl_curves(data)
    print("Done.")
