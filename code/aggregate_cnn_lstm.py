import json
import numpy as np

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

agg = {}
for fd in DATASETS:
    rmse_vals, mae_vals, phm_vals, rt_vals = [], [], [], []
    for seed in SEEDS:
        d = json.load(open(f"results/cnn_lstm/{fd}/seed_{seed}/summary_metrics.json"))
        rmse_vals.append(d["CNN_LSTM"]["RMSE"])
        mae_vals.append(d["CNN_LSTM"]["MAE"])
        phm_vals.append(d["CNN_LSTM"]["PHM_score"])
        rt_vals.append(d["_meta"]["runtime_sec"])
    rmse_vals, mae_vals, phm_vals = np.array(rmse_vals), np.array(mae_vals), np.array(phm_vals)
    agg[fd] = {
        "rmse_mean": float(rmse_vals.mean()), "rmse_std": float(rmse_vals.std(ddof=1)),
        "mae_mean": float(mae_vals.mean()), "mae_std": float(mae_vals.std(ddof=1)),
        "phm_mean": float(phm_vals.mean()), "phm_std": float(phm_vals.std(ddof=1)),
        "runtime_mean": float(np.mean(rt_vals)),
    }
    print(f"{fd}: CNN-LSTM RMSE={agg[fd]['rmse_mean']:.2f}+/-{agg[fd]['rmse_std']:.2f} "
          f"MAE={agg[fd]['mae_mean']:.2f}+/-{agg[fd]['mae_std']:.2f} "
          f"PHM={agg[fd]['phm_mean']:.1f}+/-{agg[fd]['phm_std']:.1f} "
          f"runtime={agg[fd]['runtime_mean']:.1f}s")

with open("results/cnn_lstm/aggregated_6seed.json", "w") as f:
    json.dump(agg, f, indent=2)

# Compare against plain LSTM RMSE means already known from main comparison
plain_rmse = {"FD001": 14.43, "FD002": 14.59, "FD003": 14.72, "FD004": 14.92}
print("\n=== CNN-LSTM vs Plain LSTM (raw features only) ===")
for fd in DATASETS:
    d = agg[fd]["rmse_mean"] - plain_rmse[fd]
    print(f"{fd}: CNN-LSTM {agg[fd]['rmse_mean']:.2f} vs Plain LSTM {plain_rmse[fd]:.2f} (delta={d:+.2f})")

print("\n=== LaTeX rows (append to Table III per dataset) ===")
for fd in DATASETS:
    print(f"CNN-LSTM & {agg[fd]['rmse_mean']:.2f} $\\pm$ {agg[fd]['rmse_std']:.2f} & "
          f"{agg[fd]['mae_mean']:.2f} $\\pm$ {agg[fd]['mae_std']:.2f} & "
          f"{agg[fd]['phm_mean']:.1f} $\\pm$ {agg[fd]['phm_std']:.1f} \\\\")
