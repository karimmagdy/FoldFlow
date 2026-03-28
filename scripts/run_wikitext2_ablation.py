"""Run reviewer-facing FoldFlow LM ablations on WikiText-2.

Example:
    python scripts/run_wikitext2_ablation.py --epochs 15 --seeds 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


ABLATION_MODELS = [
    "transformer++",
    "foldflow-no-energy-gate",
    "foldflow-no-chaperone",
    "foldflow-no-langevin",
    "foldflow",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run WikiText-2 FoldFlow ablation suite")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-root", type=str, default="./results/wikitext2_ablation")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.save_root)
    root.mkdir(parents=True, exist_ok=True)

    for model_name in ABLATION_MODELS:
        safe_name = model_name.replace("+", "plus").replace("/", "_")
        run_dir = root / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "scripts/train_wikitext2.py",
            "--epochs", str(args.epochs),
            "--seeds", str(args.seeds),
            "--seq-len", str(args.seq_len),
            "--batch-size", str(args.batch_size),
            "--models", model_name,
            "--save-dir", str(run_dir),
            "--results-name", f"{safe_name}.json",
        ]
        command.extend(args.extra_args)

        print(f"\n{'=' * 72}")
        print(f"Running WikiText-2 ablation entry: {model_name}")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
