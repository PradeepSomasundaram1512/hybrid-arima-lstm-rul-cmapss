import json
import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

print("Paired Wilcoxon: CNN-LSTM vs Plain LSTM, per-engine |error| averaged over 6 seeds")
for fd in DATASETS:
    cnn_err_by_seed, plain_err_by_seed = [], []
    for seed in SEEDS:
        d_cnn = json.load(open(f"results/cnn_lstm/{fd}/seed_{seed}/raw_predictions.json"))
        d_plain = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
        yt_cnn = np.array(d_cnn["CNN_LSTM"]["y_true"])
        cnn_err_by_seed.append(np.abs(np.array(d_cnn["CNN_LSTM"]["y_pred"]) - yt_cnn))
        yt_plain = np.array(d_plain["Plain_LSTM"]["y_true"])
        plain_err_by_seed.append(np.abs(np.array(d_plain["Plain_LSTM"]["y_pred"]) - yt_plain))
    cnn_mean = np.mean(cnn_err_by_seed, axis=0)
    plain_mean = np.mean(plain_err_by_seed, axis=0)
    stat, p = wilcoxon(cnn_mean, plain_mean)
    n_worse = int((cnn_mean > plain_mean).sum())
    n_total = len(cnn_mean)
    median_delta = float(np.median(cnn_mean - plain_mean))
    print(f"{fd}: p={p:.3f} CNN-LSTM worse on {n_worse}/{n_total} median_delta={median_delta:+.3f}")
