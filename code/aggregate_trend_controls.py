import json
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

agg = {}
for fd in DATASETS:
    plain_rmse, ma_rmse, ema_rmse, causal_rmse = [], [], [], []
    for seed in SEEDS:
        d = json.load(open(f"results/trend_controls/{fd}/seed_{seed}/summary_metrics.json"))
        ma_rmse.append(d["MA_Hybrid_LSTM"]["RMSE"])
        ema_rmse.append(d["EMA_Hybrid_LSTM"]["RMSE"])
        dp = json.load(open(f"results/{fd}/seed_{seed}/summary_metrics.json"))
        plain_rmse.append(dp["Plain_LSTM"]["RMSE"])
        dc = json.load(open(f"results/causal/{fd}/seed_{seed}/summary_metrics.json"))
        causal_rmse.append(dc["Hybrid_ARIMA_LSTM"]["RMSE"])
    plain_rmse, ma_rmse, ema_rmse, causal_rmse = map(np.array, (plain_rmse, ma_rmse, ema_rmse, causal_rmse))
    agg[fd] = {
        "plain_mean": float(plain_rmse.mean()), "plain_std": float(plain_rmse.std(ddof=1)),
        "ma_mean": float(ma_rmse.mean()), "ma_std": float(ma_rmse.std(ddof=1)),
        "ema_mean": float(ema_rmse.mean()), "ema_std": float(ema_rmse.std(ddof=1)),
        "causal_mean": float(causal_rmse.mean()), "causal_std": float(causal_rmse.std(ddof=1)),
    }
    print(f"{fd}: plain={agg[fd]['plain_mean']:.2f} MA={agg[fd]['ma_mean']:.2f} EMA={agg[fd]['ema_mean']:.2f} causal_ARIMA={agg[fd]['causal_mean']:.2f}")

print("\n=== per-engine paired significance tests (seed-averaged) ===")
families = {
    "MA_vs_plain": [], "EMA_vs_plain": [],
    "MA_vs_causal": [], "EMA_vs_causal": [],
}
for fd in DATASETS:
    ma_err, ema_err, plain_err, causal_err = [], [], [], []
    for seed in SEEDS:
        d = json.load(open(f"results/trend_controls/{fd}/seed_{seed}/raw_predictions.json"))
        yt_ma = np.array(d["MA_Hybrid_LSTM"]["y_true"])
        ma_err.append(np.abs(np.array(d["MA_Hybrid_LSTM"]["y_pred"]) - yt_ma))
        yt_ema = np.array(d["EMA_Hybrid_LSTM"]["y_true"])
        ema_err.append(np.abs(np.array(d["EMA_Hybrid_LSTM"]["y_pred"]) - yt_ema))
        dp = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
        ytp = np.array(dp["Plain_LSTM"]["y_true"])
        plain_err.append(np.abs(np.array(dp["Plain_LSTM"]["y_pred"]) - ytp))
        dc = json.load(open(f"results/causal/{fd}/seed_{seed}/raw_predictions.json"))
        ytc = np.array(dc["Hybrid_ARIMA_LSTM"]["y_true"])
        causal_err.append(np.abs(np.array(dc["Hybrid_ARIMA_LSTM"]["y_pred"]) - ytc))
    ma_m = np.mean(ma_err, axis=0)
    ema_m = np.mean(ema_err, axis=0)
    plain_m = np.mean(plain_err, axis=0)
    causal_m = np.mean(causal_err, axis=0)

    _, p = wilcoxon(ma_m, plain_m); families["MA_vs_plain"].append(p)
    _, p = wilcoxon(ema_m, plain_m); families["EMA_vs_plain"].append(p)
    _, p = wilcoxon(ma_m, causal_m); families["MA_vs_causal"].append(p)
    _, p = wilcoxon(ema_m, causal_m); families["EMA_vs_causal"].append(p)
    print(f"{fd}: MA vs plain p={families['MA_vs_plain'][-1]:.4f} | EMA vs plain p={families['EMA_vs_plain'][-1]:.4f} | "
          f"MA vs causal p={families['MA_vs_causal'][-1]:.4f} | EMA vs causal p={families['EMA_vs_causal'][-1]:.4f}")

print("\n=== Holm-corrected, per family ===")
for name, pvals in families.items():
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    print(f"\n{name}:")
    for fd, p, padj, rej in zip(DATASETS, pvals, p_adj, reject):
        print(f"  {fd}: raw={p:.4f} adj={padj:.4f} sig={rej}")

with open("results/trend_controls/aggregated.json", "w") as f:
    json.dump(agg, f, indent=2)
