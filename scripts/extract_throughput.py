"""Extract throughput/timing data from WikiText-2 ablation results."""
import json
import numpy as np

with open("results/wikitext2_ablation_direct/wikitext2_ablation_direct.json") as f:
    data = json.load(f)

print("=== Throughput & Timing Summary ===\n")
print(f"{'Model':<30} {'Params':>10} {'Train tok/s':>12} {'Val tok/s':>12} {'Epoch (s)':>12} {'PPL':>10}")
print("-" * 90)

for model_name, mdata in data["results"].items():
    seeds = mdata["seeds"]
    params = seeds[0]["num_parameters"]
    
    train_tps = []
    val_tps = []
    train_times = []
    
    for s in seeds:
        # Average over epochs 2-15 (skip epoch 1 warmup)
        for entry in s["log"][1:]:
            train_times.append(entry["train_epoch_time_s"])
            train_tps.append(entry["train_tokens_per_s"])
            val_tps.append(entry["val_eval_time_s"])
    
    best_ppls = [s["best_ppl"] for s in seeds]
    
    # Also get val tokens/s
    val_tokens_per_s = []
    for s in seeds:
        for entry in s["log"][1:]:
            val_tokens_per_s.append(entry["val_tokens_per_s"])
    
    print(f"{model_name:<30} {params/1e6:>9.1f}M {np.mean(train_tps):>11,.0f} {np.mean(val_tokens_per_s):>11,.0f} {np.mean(train_times):>11,.0f} {np.mean(best_ppls):>9.1f}")

print()
print("=== Relative Throughput (vs Transformer++) ===")
# Compute relative to transformer++
t_tps = []
for s in data["results"]["transformer++"]["seeds"]:
    for entry in s["log"][1:]:
        t_tps.append(entry["train_tokens_per_s"])
t_mean = np.mean(t_tps)

for model_name, mdata in data["results"].items():
    tps = []
    for s in mdata["seeds"]:
        for entry in s["log"][1:]:
            tps.append(entry["train_tokens_per_s"])
    ratio = np.mean(tps) / t_mean
    print(f"  {model_name}: {ratio:.2f}x")
