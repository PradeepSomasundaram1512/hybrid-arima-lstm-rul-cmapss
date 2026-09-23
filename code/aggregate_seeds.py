"""
Aggregate multi-seed results (3 seeds x 4 datasets) into:
  - mean +/- std RMSE/MAE/PHM per dataset per model
  - a properly-powered paired significance test using per-engine errors
    AVERAGED across seeds (reduces per-engine noise vs a single-seed test)
"""
import json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]
MODELS = ["RandomForest_baseline", "Plain_LSTM", "Hybrid_ARIMA_LSTM"]

agg = {}
sig = {}

for fd in DATASETS:
    agg[fd] = {}
    per_model_seed_metrics = {m: {"RMSE": [], "MAE": [], "PHM_score": []} for m in MODELS}

    # for the averaged-per-engine significance test: engine order must match across seeds
    plain_err_by_seed = []
    hybrid_err_by_seed = []
    yt_ref = None

    for seed in SEEDS:
        with open(f"results/{fd}/seed_{seed}/summary_metrics.json") as f:
            summary = json.load(f)
        for m in MODELS:
            per_model_seed_metrics[m]["RMSE"].append(summary[m]["RMSE"])
            per_model_seed_metrics[m]["MAE"].append(summary[m]["MAE"])
            per_model_seed_metrics[m]["PHM_score"].append(summary[m]["PHM_score"])

        with open(f"results/{fd}/seed_{seed}/raw_predictions.json") as f:
            preds = json.load(f)
        yt = np.array(preds["Plain_LSTM"]["y_true"])
        if yt_ref is None:
            yt_ref = yt
        # engine identity/order is stable across seeds for a fixed test set (sorted unit numbers),
        # only the model weights/val-split differ by seed, so this alignment is valid
        plain_err_by_seed.append(np.abs(np.array(preds["Plain_LSTM"]["y_pred"]) - yt))
        hybrid_err_by_seed.append(np.abs(np.array(preds["Hybrid_ARIMA_LSTM"]["y_pred"]) - yt))

    for m in MODELS:
        rmse_arr = np.array(per_model_seed_metrics[m]["RMSE"])
        mae_arr = np.array(per_model_seed_metrics[m]["MAE"])
        phm_arr = np.array(per_model_seed_metrics[m]["PHM_score"])
        agg[fd][m] = {
            "RMSE_mean": float(rmse_arr.mean()), "RMSE_std": float(rmse_arr.std(ddof=1)),
            "MAE_mean": float(mae_arr.mean()), "MAE_std": float(mae_arr.std(ddof=1)),
            "PHM_mean": float(phm_arr.mean()), "PHM_std": float(phm_arr.std(ddof=1)),
            "per_seed_RMSE": rmse_arr.tolist(),
        }

    # per-engine error averaged across the 3 seeds, then paired Wilcoxon (n = n_engines)
    plain_err_mean = np.mean(plain_err_by_seed, axis=0)
    hybrid_err_mean = np.mean(hybrid_err_by_seed, axis=0)
    stat, p = wilcoxon(hybrid_err_mean, plain_err_mean)
    diff = hybrid_err_mean - plain_err_mean
    sig[fd] = {
        "wilcoxon_stat": float(stat), "p_value": float(p),
        "n_hybrid_better": int((diff < 0).sum()), "n_plain_better": int((diff > 0).sum()),
        "n_engines": len(yt_ref), "n_seeds_averaged": len(SEEDS),
        "median_diff_abs_error": float(np.median(diff)),
    }
    print(f"{fd}: seed-averaged Wilcoxon p={p:.4f} | hybrid better on {sig[fd]['n_hybrid_better']}/{sig[fd]['n_engines']} engines")

with open("results/aggregated_multiseed.json", "w") as f:
    json.dump({"agg": agg, "significance": sig}, f, indent=2)

print("\n=== Mean +/- std RMSE per dataset per model (3 seeds) ===")
for fd in DATASETS:
    for m in MODELS:
        a = agg[fd][m]
        print(f"{fd} {m}: RMSE={a['RMSE_mean']:.2f}+/-{a['RMSE_std']:.2f}  "
              f"MAE={a['MAE_mean']:.2f}+/-{a['MAE_std']:.2f}  PHM={a['PHM_mean']:.1f}+/-{a['PHM_std']:.1f}")

# ---- LaTeX table ----
print("\n=== LaTeX table ===")
print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\caption{RUL prediction performance across NASA C-MAPSS sub-datasets, mean $\pm$ std over 3 seeds (42, 123, 2024).}")
print(r"\label{tab:main_results}")
print(r"\begin{tabular}{llccc}")
print(r"\toprule")
print(r"Dataset & Model & RMSE & MAE & PHM Score \\")
print(r"\midrule")
labels = {"RandomForest_baseline": "Random Forest", "Plain_LSTM": "Plain LSTM", "Hybrid_ARIMA_LSTM": "Hybrid ARIMA-LSTM (proposed)"}
ds_labels = {"FD001": "FD001 (1 condition, 1 fault)", "FD002": "FD002 (6 conditions, 1 fault)",
             "FD003": "FD003 (1 condition, 2 faults)", "FD004": "FD004 (6 conditions, 2 faults)"}
for fd in DATASETS:
    for i, m in enumerate(MODELS):
        a = agg[fd][m]
        dsname = ds_labels[fd] if i == 0 else ""
        print(f"{dsname} & {labels[m]} & {a['RMSE_mean']:.2f} $\\pm$ {a['RMSE_std']:.2f} & "
              f"{a['MAE_mean']:.2f} $\\pm$ {a['MAE_std']:.2f} & {a['PHM_mean']:.1f} $\\pm$ {a['PHM_std']:.1f} \\\\")
    print(r"\midrule" if fd != DATASETS[-1] else r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")

print("\n=== Significance table (LaTeX) ===")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Paired Wilcoxon test on per-engine absolute error, averaged over 3 seeds (hybrid vs.\ plain LSTM).}")
print(r"\label{tab:wilcoxon}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Dataset & $p$-value & Hybrid better on & Median $\Delta|$err$|$ \\")
print(r"\midrule")
for fd in DATASETS:
    s = sig[fd]
    print(f"{fd} & {s['p_value']:.3f} & {s['n_hybrid_better']}/{s['n_engines']} engines & {s['median_diff_abs_error']:+.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
