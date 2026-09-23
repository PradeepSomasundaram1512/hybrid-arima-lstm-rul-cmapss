import json
import numpy as np

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

agg = {}
for fd in DATASETS:
    rmse_vals, mae_vals, phm_vals, gate_vals = [], [], [], []
    for seed in SEEDS:
        d = json.load(open(f"results/gated_fusion/{fd}/seed_{seed}/summary_metrics.json"))
        rmse_vals.append(d["Gated_Fusion"]["RMSE"])
        mae_vals.append(d["Gated_Fusion"]["MAE"])
        phm_vals.append(d["Gated_Fusion"]["PHM_score"])
        gate_vals.append(d["_meta"]["mean_test_gate_value"])
    rmse_vals, mae_vals, phm_vals = np.array(rmse_vals), np.array(mae_vals), np.array(phm_vals)
    agg[fd] = {
        "rmse_mean": float(rmse_vals.mean()), "rmse_std": float(rmse_vals.std(ddof=1)),
        "mae_mean": float(mae_vals.mean()), "mae_std": float(mae_vals.std(ddof=1)),
        "phm_mean": float(phm_vals.mean()), "phm_std": float(phm_vals.std(ddof=1)),
        "mean_gate": float(np.mean(gate_vals)),
    }
    print(f"{fd}: Gated-Fusion RMSE={agg[fd]['rmse_mean']:.2f}+/-{agg[fd]['rmse_std']:.2f} "
          f"MAE={agg[fd]['mae_mean']:.2f}+/-{agg[fd]['mae_std']:.2f} "
          f"PHM={agg[fd]['phm_mean']:.1f}+/-{agg[fd]['phm_std']:.1f} "
          f"mean_gate={agg[fd]['mean_gate']:.3f}")

with open("results/gated_fusion/aggregated_6seed.json", "w") as f:
    json.dump(agg, f, indent=2)

plain = {"FD001": 14.43, "FD002": 14.59, "FD003": 14.72, "FD004": 14.92}
fixed_hybrid = {"FD001": 14.92, "FD002": 14.93, "FD003": 14.87, "FD004": 15.71}
selected_hybrid = {"FD001": 14.38, "FD002": 13.77, "FD003": 14.45, "FD004": 14.90}

print("\n=== Gated Fusion vs all prior models (RMSE, 6-seed means) ===")
print(f"{'Dataset':8s} {'Plain':>8s} {'Fixed-H':>8s} {'Sel-H (best)':>13s} {'Gated':>8s} {'vs best':>10s}")
for fd in DATASETS:
    g = agg[fd]["rmse_mean"]
    best = selected_hybrid[fd]
    delta = g - best
    print(f"{fd:8s} {plain[fd]:8.2f} {fixed_hybrid[fd]:8.2f} {best:13.2f} {g:8.2f} {delta:+10.2f}")

print("\n=== LaTeX table row ===")
for fd in DATASETS:
    print(f"{fd} & {plain[fd]:.2f} & {fixed_hybrid[fd]:.2f} & {selected_hybrid[fd]:.2f} & "
          f"{agg[fd]['rmse_mean']:.2f} $\\pm$ {agg[fd]['rmse_std']:.2f} \\\\")
