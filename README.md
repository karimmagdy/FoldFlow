# FoldFlow: Neural Architecture Inspired by Protein Folding

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20932167.svg)](https://doi.org/10.5281/zenodo.20932167)

Codebase for the paper **FoldFlow: A Systematic Study of Bio-Inspired Inductive Biases for Neural Architecture Design**.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision datasets transformers timm

# Smoke test (30 epochs CIFAR-10)
python scripts/train_cifar10.py --quick

# Full training (200 epochs)
python scripts/train_cifar10.py --epochs 200 --encoder weak

# Ablation study (7 configs × 3 seeds)
python scripts/run_ablation.py --epochs 100

# WikiText-2 comparison (fair: same hyperparams for all)
python scripts/train_wikitext2.py --epochs 10 --seeds 3

# Generate paper figures from finished runs
python scripts/generate_figures.py --results-dir results --save-dir figures
```

## Final Results

### CIFAR-10

- FoldFlow classifier (weak encoder, 200 epochs): **85.72%** best top-1 accuracy
- Parameters: **1,480,824**
- Final metrics stored in `checkpoints/cifar10_results.json`

### WikiText-2

Fair comparison with identical hyperparameters for both models (`d_model=512`, `n_layers=6`, `seq_len=256`, `batch_size=16`, `epochs=10`, `seeds=3`):

| Model | Perplexity (mean ± std) | Parameters |
|---|---:|---:|
| Transformer++ | 594.3 ± 2.6 | 44,777,984 |
| FoldFlow LM | **515.9 ± 3.6** | 50,166,071 |

- FoldFlow improves perplexity by **13.2%** relative to Transformer++
- Final metrics stored in `results/wikitext2_results.json`

## Generated Figures

- `figures/fig1_ablation.pdf`: ablation results across FoldFlow characteristics
- `figures/fig2_training_curves.pdf`: CIFAR-10 training curves
- `figures/fig3_wikitext2.pdf`: WikiText-2 perplexity comparison

## Reproducibility Notes

- CIFAR-10 full training can be resumed from a checkpoint:

```bash
python scripts/train_cifar10.py --epochs 200 --encoder weak --resume checkpoints/foldflow_cifar10_best.pt
```

- WikiText-2 loading uses the local Hugging Face cache to avoid remote download stalls during tokenizer and dataset initialization.

## Project Structure

```
FolwFlowRelease/
├── foldflow/
│   ├── models/
│   │   ├── classifier.py    # FoldFlow image classifier (5 characteristics)
│   │   └── lm.py            # FoldFlow LM + Transformer++ baseline
│   ├── dynamics/
│   │   └── langevin.py      # Core Langevin dynamics with stress signal
│   ├── data/
│   │   ├── cifar10.py       # CIFAR-10 loading + CutMix/MixUp
│   │   └── wikitext2.py     # WikiText-2 tokenization
│   └── utils/
│       └── training.py      # Metrics, EMA, cosine schedule
├── scripts/
│   ├── train_cifar10.py     # Full CIFAR-10 training
│   ├── run_ablation.py      # Ablation experiments
│   └── train_wikitext2.py   # Fair WikiText-2 comparison
├── results/                 # JSON results
├── checkpoints/             # Model weights
└── figures/                 # Paper figures
```

## The 5 FoldFlow Characteristics

| # | Characteristic | Description | Implementation |
|---|---------------|-------------|----------------|
| C1 | Energy Minimization | Langevin dynamics finds equilibrium | `dynamics/langevin.py` |
| C2 | Dynamic Topology | Input-conditioned connections | Multi-head attention |
| C3 | Cooperativity | Local interactions → global structure | 1D depthwise conv |
| C4 | Chaperone | Stress-gated guided folding | `||∇E||`-gated correction |
| C5 | Environmental Sensitivity | Context-dependent modulation | Scale + shift |

## Key Fixes Over v1

1. **Cooperativity**: Replaced 2-layer MHA+FFN (harmful at small scale) with lightweight 1D depthwise convolution for true local particle interactions.

2. **Chaperone**: Implemented stress-gated intervention using actual `||∇E||` from Langevin dynamics (paper's described mechanism). Old code used broken `0.1 * softmax` scaling.

3. **Training**: Full 50K CIFAR-10 (was 25K), 200 epochs (was 30), CutMix/MixUp/AutoAugment, label smoothing, cosine warmup, EMA.

4. **Fair baselines**: WikiText-2 uses IDENTICAL hyperparameters for all models (same d_model, n_layers, seq_len, batch_size, lr, epochs).

## Ablation Configs

| Config | C1 | C2 | C3 | C4 | C5 |
|--------|----|----|----|----|-----|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| +Energy | ✓ | ✗ | ✗ | ✗ | ✗ |
| +Energy+Topology | ✓ | ✓ | ✗ | ✗ | ✗ |
| +Energy+Cooperativity | ✓ | ✗ | ✓ | ✗ | ✗ |
| +Energy+Chaperone | ✓ | ✗ | ✗ | ✓ | ✗ |
| +Energy+EnvSensitivity | ✓ | ✗ | ✗ | ✗ | ✓ |
| Full Model | ✓ | ✓ | ✓ | ✓ | ✓ |

## Hardware Requirements

- **Minimum**: GPU with 6GB VRAM (RTX 3050)
- **Recommended**: GPU with 8GB+ VRAM or Apple M-series with 16GB+
- CIFAR-10 full training: ~2-4 hours on RTX 3050
- Ablation study (21 runs): ~10-20 hours on RTX 3050
- WikiText-2 (2 models × 3 seeds): ~3-6 hours on RTX 3050
