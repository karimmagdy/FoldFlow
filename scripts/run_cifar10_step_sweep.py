"""Run a CIFAR-10 sweep over FoldFlow dynamics depth.

Example:
    python scripts/run_cifar10_step_sweep.py --steps 0 2 4 8 --epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 step-depth sweep")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seeds to run (e.g. --seeds 42 43 44)")
    parser.add_argument("--encoder", type=str, default="weak", choices=["weak", "strong"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-root", type=str, default="./results/cifar10_step_sweep")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.save_root)
    root.mkdir(parents=True, exist_ok=True)

    for step_count in args.steps:
        for seed in args.seeds:
            run_dir = root / f"steps_{step_count}"
            run_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                "scripts/train_cifar10.py",
                "--epochs", str(args.epochs),
                "--encoder", args.encoder,
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--num-steps", str(step_count),
                "--seed", str(seed),
                "--save-dir", str(run_dir),
                "--results-name", f"cifar10_steps_{step_count}_seed{seed}.json",
                "--checkpoint-name", f"foldflow_steps_{step_count}_seed{seed}_best.pt",
            ]
            command.extend(args.extra_args)

            print(f"\n{'=' * 72}")
            print(f"Running CIFAR-10 sweep: num_steps={step_count}, seed={seed}")
            print(" ".join(command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
