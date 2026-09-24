import json
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

agg = {}
for fd in DATASETS:
    plain_rmse, causal_rmse, noncausal_rmse = [], [], []
    missing = []
    for seed in SEEDS:
        try:
            d = json.load(open(f"results/causal/{fd}/seed_{seed}/summary_metrics.json"))
        except FileNotFoundError:
            missing.append(seed)
            continue
        causal_rmse.append(d["Hybrid_ARIMA_LSTM"]["RMSE"])
        plain_rmse.append(d["Plain_LSTM"]["RMSE"])
        d2 = json.load(open(f"results/{fd}/seed_{seed}/summary_metrics.json"))
        noncausal_rmse.append(d2["Hybrid_ARIMA_LSTM"]["RMSE"])
    n = len(causal_rmse)
    if missing:
        print(f"{fd}: INCOMPLETE, missing seeds {missing}, have {n}/6")
    if n == 0:
        continue
    plain_rmse, causal_rmse, noncausal_rmse = map(np.array, (plain_rmse, causal_rmse, noncausal_rmse))
    agg[fd] = {
        "n_seeds": n,
        "plain_mean": float(plain_rmse.mean()), "plain_std": float(plain_rmse.std(ddof=1)) if n > 1 else None,
        "causal_mean": float(causal_rmse.mean()), "causal_std": float(causal_rmse.std(ddof=1)) if n > 1 else None,
        "noncausal_mean": float(noncausal_rmse.mean()), "noncausal_std": float(noncausal_rmse.std(ddof=1)) if n > 1 else None,
    }
    print(f"{fd} (n={n} seeds): plain={agg[fd]['plain_mean']:.2f} noncausal_hybrid={agg[fd]['noncausal_mean']:.2f} causal_hybrid={agg[fd]['causal_mean']:.2f}")

if all(fd in agg and agg[fd]["n_seeds"] == 6 for fd in DATASETS):
    print("\n=== ALL 24 RUNS COMPLETE: per-engine paired significance tests ===")
    pvals_vs_plain, pvals_vs_noncausal = [], []
    for fd in DATASETS:
        causal_err, plain_err, noncausal_err = [], [], []
        for seed in SEEDS:
            dc = json.load(open(f"results/causal/{fd}/seed_{seed}/raw_predictions.json"))
            ytc = np.array(dc["Hybrid_ARIMA_LSTM"]["y_true"])
            causal_err.append(np.abs(np.array(dc["Hybrid_ARIMA_LSTM"]["y_pred"]) - ytc))
            plain_err.append(np.abs(np.array(dc["Plain_LSTM"]["y_pred"]) - np.array(dc["Plain_LSTM"]["y_true"])))
            dn = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
            ytn = np.array(dn["Hybrid_ARIMA_LSTM"]["y_true"])
            noncausal_err.append(np.abs(np.array(dn["Hybrid_ARIMA_LSTM"]["y_pred"]) - ytn))
        causal_m = np.mean(causal_err, axis=0)
        plain_m = np.mean(plain_err, axis=0)
        noncausal_m = np.mean(noncausal_err, axis=0)
        _, p_vs_plain = wilcoxon(causal_m, plain_m)
        _, p_vs_noncausal = wilcoxon(causal_m, noncausal_m)
        pvals_vs_plain.append(p_vs_plain)
        pvals_vs_noncausal.append(p_vs_noncausal)
        print(f"{fd}: causal vs plain p={p_vs_plain:.4f} | causal vs non-causal p={p_vs_noncausal:.4f}")

    print("\nHolm-corrected, causal hybrid vs plain LSTM:")
    reject, p_adj, _, _ = multipletests(pvals_vs_plain, alpha=0.05, method="holm")
    for fd, p, padj, rej in zip(DATASETS, pvals_vs_plain, p_adj, reject):
        print(f"  {fd}: raw={p:.4f} adj={padj:.4f} sig={rej}")

    print("\nHolm-corrected, causal hybrid vs non-causal hybrid:")
    reject, p_adj, _, _ = multipletests(pvals_vs_noncausal, alpha=0.05, method="holm")
    for fd, p, padj, rej in zip(DATASETS, pvals_vs_noncausal, p_adj, reject):
        print(f"  {fd}: raw={p:.4f} adj={padj:.4f} sig={rej}")
else:
    print("\n=== NOT YET COMPLETE: significance tests withheld until all 24 runs finish ===")

with open("results/causal/aggregated.json", "w") as f:
    json.dump(agg, f, indent=2)
