import json
import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

print("Paired Wilcoxon: Gated Fusion vs Plain LSTM, per-engine |error| averaged over 6 seeds")
for fd in DATASETS:
    gated_err_by_seed, plain_err_by_seed = [], []
    for seed in SEEDS:
        d_gated = json.load(open(f"results/gated_fusion/{fd}/seed_{seed}/raw_predictions.json"))
        d_plain = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
        yt_gated = np.array(d_gated["Gated_Fusion"]["y_true"])
        gated_err_by_seed.append(np.abs(np.array(d_gated["Gated_Fusion"]["y_pred"]) - yt_gated))
        yt_plain = np.array(d_plain["Plain_LSTM"]["y_true"])
        plain_err_by_seed.append(np.abs(np.array(d_plain["Plain_LSTM"]["y_pred"]) - yt_plain))
    gated_mean = np.mean(gated_err_by_seed, axis=0)
    plain_mean = np.mean(plain_err_by_seed, axis=0)
    stat, p = wilcoxon(gated_mean, plain_mean)
    n_worse = int((gated_mean > plain_mean).sum())
    n_total = len(gated_mean)
    median_delta = float(np.median(gated_mean - plain_mean))
    print(f"{fd}: p={p:.3f} Gated worse on {n_worse}/{n_total} median_delta={median_delta:+.3f}")

print("\nPaired Wilcoxon: Gated Fusion vs Selected-Order Hybrid, per-engine |error| averaged over 6 seeds")
for fd in DATASETS:
    gated_err_by_seed, sel_err_by_seed = [], []
    for seed in SEEDS:
        d_gated = json.load(open(f"results/gated_fusion/{fd}/seed_{seed}/raw_predictions.json"))
        d_sel = json.load(open(f"results/order_selected/{fd}/seed_{seed}/raw_predictions.json"))
        yt_gated = np.array(d_gated["Gated_Fusion"]["y_true"])
        gated_err_by_seed.append(np.abs(np.array(d_gated["Gated_Fusion"]["y_pred"]) - yt_gated))
        yt_sel = np.array(d_sel["Hybrid_ARIMA_LSTM"]["y_true"])
        sel_err_by_seed.append(np.abs(np.array(d_sel["Hybrid_ARIMA_LSTM"]["y_pred"]) - yt_sel))
    gated_mean = np.mean(gated_err_by_seed, axis=0)
    sel_mean = np.mean(sel_err_by_seed, axis=0)
    stat, p = wilcoxon(gated_mean, sel_mean)
    n_worse = int((gated_mean > sel_mean).sum())
    n_total = len(gated_mean)
    median_delta = float(np.median(gated_mean - sel_mean))
    print(f"{fd}: p={p:.3f} Gated worse on {n_worse}/{n_total} median_delta={median_delta:+.3f}")
