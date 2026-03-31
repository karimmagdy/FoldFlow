"""Run statistical tests on FoldFlow experiment results.

Reads results JSON files, runs paired t-tests and bootstrap CIs for all
ablation comparisons, and outputs formatted tables suitable for LaTeX.

Usage:
    python scripts/run_statistical_tests.py
    python scripts/run_statistical_tests.py --results results/wikitext2_convergence.json
    python scripts/run_statistical_tests.py --results results/wikitext2_convergence.json --baseline transformer++
    python scripts/run_statistical_tests.py --results results/ablation_results.json --metric best_accuracy
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


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


def bootstrap_diff_ci(a, b, n_bootstrap=10000, ci=0.95, rng=None):
    """Bootstrap CI for the difference of means (a - b)."""
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    boot_diffs = []
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a, size=len(a), replace=True)
        b_sample = rng.choice(b, size=len(b), replace=True)
        boot_diffs.append(a_sample.mean() - b_sample.mean())
    boot_diffs = np.array(boot_diffs)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_diffs, 100 * alpha))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha)))
    return float(a.mean() - b.mean()), lo, hi


def paired_ttest(a, b):
    """Paired t-test (two-sided). Returns t-statistic and p-value.

    Uses scipy if available, otherwise falls back to manual computation.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(a, b)
        return float(t_stat), float(p_value)
    except ImportError:
        # Manual paired t-test
        diff = a - b
        n = len(diff)
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)
        if std_diff == 0 or n < 2:
            return 0.0, 1.0
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        # Two-sided p-value approximation using normal (conservative for small n)
        from math import erfc, sqrt
        p_value = erfc(abs(t_stat) / sqrt(2))
        return float(t_stat), float(p_value)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def extract_seed_values(results: dict, metric: str) -> dict:
    """Extract per-seed metric values from a results dict.

    Supports both wikitext2 format (best_ppl) and ablation format (best_accuracy).
    Returns {model_name: [val_seed0, val_seed1, ...]}.
    """
    model_values = {}
    for model_name, model_data in results.items():
        seeds = model_data.get("seeds", [])
        values = []
        for s in seeds:
            if metric in s:
                values.append(s[metric])
            elif metric == "best_ppl" and "best_accuracy" in s:
                # Wrong metric for this file type
                continue
            elif metric == "best_accuracy" and "best_ppl" in s:
                continue
        if values:
            model_values[model_name] = values
    return model_values


def detect_metric(results: dict) -> str:
    """Auto-detect the primary metric from the results structure."""
    for model_data in results.values():
        seeds = model_data.get("seeds", [])
        if seeds:
            if "best_ppl" in seeds[0]:
                return "best_ppl"
            if "best_accuracy" in seeds[0]:
                return "best_accuracy"
    return "best_ppl"


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def run_comparisons(model_values: dict, baseline_name: str, metric: str):
    """Run pairwise comparisons: each model vs baseline, plus all pairs.

    Returns a list of comparison dicts.
    """
    rng = np.random.default_rng(42)
    comparisons = []

    models = list(model_values.keys())

    # 1) Each model vs baseline
    if baseline_name in model_values:
        baseline_vals = model_values[baseline_name]
        for name in models:
            if name == baseline_name:
                continue
            vals = model_values[name]

            # Paired t-test (only if same number of seeds)
            if len(vals) == len(baseline_vals):
                t_stat, p_value = paired_ttest(vals, baseline_vals)
            else:
                t_stat, p_value = None, None

            # Bootstrap CI on the difference
            diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(
                vals, baseline_vals, rng=rng,
            )

            # Bootstrap CI on each
            mean_a, ci_lo_a, ci_hi_a = bootstrap_ci(vals, rng=rng)
            mean_b, ci_lo_b, ci_hi_b = bootstrap_ci(baseline_vals, rng=rng)

            comparisons.append({
                "model_a": name,
                "model_b": baseline_name,
                "metric": metric,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "ci_95_a": [ci_lo_a, ci_hi_a],
                "ci_95_b": [ci_lo_b, ci_hi_b],
                "diff_mean": diff_mean,
                "diff_ci_95": [diff_lo, diff_hi],
                "t_statistic": t_stat,
                "p_value": p_value,
                "n_seeds_a": len(vals),
                "n_seeds_b": len(baseline_vals),
                "significant_005": p_value < 0.05 if p_value is not None else None,
            })

    # 2) All pairwise (excluding baseline comparisons already done)
    done_pairs = {(c["model_a"], c["model_b"]) for c in comparisons}
    for a_name, b_name in combinations(models, 2):
        if (a_name, b_name) in done_pairs or (b_name, a_name) in done_pairs:
            continue
        a_vals = model_values[a_name]
        b_vals = model_values[b_name]

        if len(a_vals) == len(b_vals):
            t_stat, p_value = paired_ttest(a_vals, b_vals)
        else:
            t_stat, p_value = None, None

        diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(a_vals, b_vals, rng=rng)

        comparisons.append({
            "model_a": a_name,
            "model_b": b_name,
            "metric": metric,
            "mean_a": float(np.mean(a_vals)),
            "mean_b": float(np.mean(b_vals)),
            "diff_mean": diff_mean,
            "diff_ci_95": [diff_lo, diff_hi],
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_seeds_a": len(a_vals),
            "n_seeds_b": len(b_vals),
            "significant_005": p_value < 0.05 if p_value is not None else None,
        })

    return comparisons


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary_table(model_values: dict, metric: str):
    """Print a summary table of all models."""
    rng = np.random.default_rng(42)
    metric_label = "PPL" if "ppl" in metric else "Acc (%)"
    print(f"\n{'='*72}")
    print(f"{'Model':<30} {metric_label:>10} {'95% CI':>20} {'Seeds':>6}")
    print(f"{'='*72}")
    for name, vals in model_values.items():
        mean, ci_lo, ci_hi = bootstrap_ci(vals, rng=rng)
        std = np.std(vals)
        print(
            f"{name:<30} "
            f"{mean:>7.2f} +/- {std:.2f} "
            f"[{ci_lo:.2f}, {ci_hi:.2f}] "
            f"{len(vals):>4}"
        )
    print(f"{'='*72}")


def print_comparison_table(comparisons: list):
    """Print pairwise comparison table."""
    print(f"\n{'='*95}")
    print(
        f"{'Model A':<24} {'Model B':<24} "
        f"{'Diff':>8} {'95% CI Diff':>18} {'t':>7} {'p':>8} {'Sig':>5}"
    )
    print(f"{'='*95}")
    for c in comparisons:
        t_str = f"{c['t_statistic']:.3f}" if c["t_statistic"] is not None else "  N/A"
        p_str = f"{c['p_value']:.4f}" if c["p_value"] is not None else "   N/A"
        sig_str = "*" if c.get("significant_005") else ""
        print(
            f"{c['model_a']:<24} {c['model_b']:<24} "
            f"{c['diff_mean']:>+8.2f} "
            f"[{c['diff_ci_95'][0]:>+.2f}, {c['diff_ci_95'][1]:>+.2f}] "
            f"{t_str:>7} {p_str:>8} {sig_str:>5}"
        )
    print(f"{'='*95}")


def generate_latex_table(comparisons: list, model_values: dict, metric: str) -> str:
    """Generate a LaTeX table for the paper."""
    rng = np.random.default_rng(42)
    metric_label = "PPL" if "ppl" in metric else "Accuracy (\\%)"
    lower_is_better = "ppl" in metric

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{WikiText-2 convergence results ({metric_label}).}}")
    lines.append("\\label{tab:convergence}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append(
        f"Model & {metric_label} & 95\\% CI & $p$-value & Sig. \\\\"
    )
    lines.append("\\midrule")

    # Find best model
    best_name = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for name, vals in model_values.items():
        m = np.mean(vals)
        if (lower_is_better and m < best_val) or (not lower_is_better and m > best_val):
            best_val = m
            best_name = name

    for name, vals in model_values.items():
        mean, ci_lo, ci_hi = bootstrap_ci(vals, rng=rng)
        std = np.std(vals)

        # Find p-value vs baseline (first comparison involving this model)
        p_str = "---"
        sig_str = ""
        for c in comparisons:
            if c["model_a"] == name or c["model_b"] == name:
                if c["p_value"] is not None:
                    p_str = f"{c['p_value']:.4f}"
                    if c["p_value"] < 0.05:
                        sig_str = "$^*$"
                break

        bold = "\\textbf" if name == best_name else ""
        if bold:
            val_str = f"\\textbf{{{mean:.1f} $\\pm$ {std:.1f}}}"
        else:
            val_str = f"{mean:.1f} $\\pm$ {std:.1f}"

        lines.append(
            f"  {name} & {val_str} & [{ci_lo:.1f}, {ci_hi:.1f}] "
            f"& {p_str} & {sig_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical tests for FoldFlow experiment results"
    )
    parser.add_argument(
        "--results", type=str, nargs="+",
        default=["./results/wikitext2_convergence.json"],
        help="Path(s) to results JSON file(s)",
    )
    parser.add_argument(
        "--metric", type=str, default=None,
        help="Metric to compare (auto-detected if not given). "
             "E.g., best_ppl, best_accuracy",
    )
    parser.add_argument(
        "--baseline", type=str, default="transformer++",
        help="Name of baseline model for pairwise comparisons",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results",
        help="Directory to save output JSON",
    )
    parser.add_argument(
        "--save-name", type=str, default="statistical_tests.json",
        help="Filename for output JSON",
    )
    parser.add_argument(
        "--latex", action="store_true", default=True,
        help="Print LaTeX table (default: True)",
    )
    parser.add_argument(
        "--no-latex", action="store_false", dest="latex",
        help="Suppress LaTeX table output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge all results files
    merged_results = {}
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        merged_results.update(results)
        print(f"Loaded {len(results)} models from {path}")

    if not merged_results:
        print("Error: No results loaded. Check --results paths.")
        sys.exit(1)

    # Detect or validate metric
    metric = args.metric or detect_metric(merged_results)
    print(f"Metric: {metric}")

    # Extract per-seed values
    model_values = extract_seed_values(merged_results, metric)
    if not model_values:
        print(f"Error: No seed-level '{metric}' values found in results.")
        sys.exit(1)

    print(f"Models found: {list(model_values.keys())}")
    print(f"Seeds per model: {[len(v) for v in model_values.values()]}")

    # Summary table
    print_summary_table(model_values, metric)

    # Run comparisons
    comparisons = run_comparisons(model_values, args.baseline, metric)

    # Print comparison table
    print_comparison_table(comparisons)

    # LaTeX table
    if args.latex:
        latex = generate_latex_table(comparisons, model_values, metric)
        print(f"\n--- LaTeX Table ---\n{latex}\n")

    # Save to JSON
    save_path = Path(args.save_dir) / args.save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metric": metric,
        "baseline": args.baseline,
        "source_files": args.results,
        "summary": {},
        "comparisons": comparisons,
    }

    rng = np.random.default_rng(42)
    for name, vals in model_values.items():
        mean, ci_lo, ci_hi = bootstrap_ci(vals, rng=rng)
        output["summary"][name] = {
            "mean": mean,
            "std": float(np.std(vals)),
            "ci_95": [ci_lo, ci_hi],
            "n_seeds": len(vals),
            "per_seed": vals,
        }

    if args.latex:
        output["latex_table"] = latex

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
