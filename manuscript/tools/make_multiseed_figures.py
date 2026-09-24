import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]
MODELS = ["RandomForest_baseline", "Plain_LSTM", "Hybrid_ARIMA_LSTM"]
labels = {"RandomForest_baseline": "Random Forest", "Plain_LSTM": "Plain LSTM", "Hybrid_ARIMA_LSTM": "Hybrid ARIMA-LSTM"}
colors = {"RandomForest_baseline": "#94a3b8", "Plain_LSTM": "#1f6feb", "Hybrid_ARIMA_LSTM": "#e2662f"}

agg = json.load(open("results/aggregated_multiseed.json"))["agg"]

# ---- Fig 1: cross-dataset RMSE, mean +/- std over 3 seeds ----
fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = np.arange(len(DATASETS))
width = 0.25
for i, m in enumerate(MODELS):
    means = [agg[fd][m]["RMSE_mean"] for fd in DATASETS]
    stds = [agg[fd][m]["RMSE_std"] for fd in DATASETS]
    ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=labels[m], color=colors[m])
ax.set_xticks(x)
ax.set_xticklabels(DATASETS)
ax.set_ylabel("RMSE (cycles)")
ax.set_title("RUL prediction RMSE, mean ± std over 6 seeds")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("manuscript/icaiet2027/figures/fig_cross_dataset.pdf", bbox_inches="tight")
plt.savefig("manuscript/icaiet2027/figures/fig_cross_dataset.png", dpi=150, bbox_inches="tight")
print("Saved fig_cross_dataset (multi-seed)")

# ---- Fig 2: predicted vs true RUL scatter, hybrid model, pooled across 3 seeds ----
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
for ax, fd in zip(axes, DATASETS):
    all_yt, all_yp = [], []
    for seed in SEEDS:
        preds = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
        r = preds["Hybrid_ARIMA_LSTM"]
        all_yt.extend(r["y_true"])
        all_yp.extend(r["y_pred"])
    yt, yp = np.array(all_yt), np.array(all_yp)
    ax.scatter(yt, yp, s=8, alpha=0.35, color="#e2662f", edgecolor="none")
    lims = [0, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(fd, fontsize=10)
    ax.set_xlabel("True RUL")
axes[0].set_ylabel("Predicted RUL")
plt.tight_layout()
plt.savefig("manuscript/icaiet2027/figures/fig_scatter_all.pdf", bbox_inches="tight")
plt.savefig("manuscript/icaiet2027/figures/fig_scatter_all.png", dpi=150, bbox_inches="tight")
print("Saved fig_scatter_all (pooled 3 seeds, hybrid model)")

# ---- Fig 3: per-engine absolute error, seed-averaged, plain vs hybrid ----
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), sharey=True)
for ax, fd in zip(axes, DATASETS):
    plain_errs, hybrid_errs = [], []
    for seed in SEEDS:
        preds = json.load(open(f"results/{fd}/seed_{seed}/raw_predictions.json"))
        yt = np.array(preds["Plain_LSTM"]["y_true"])
        plain_errs.append(np.abs(np.array(preds["Plain_LSTM"]["y_pred"]) - yt))
        hybrid_errs.append(np.abs(np.array(preds["Hybrid_ARIMA_LSTM"]["y_pred"]) - yt))
    plain_mean = np.mean(plain_errs, axis=0)
    hybrid_mean = np.mean(hybrid_errs, axis=0)
    bp = ax.boxplot([plain_mean, hybrid_mean], tick_labels=["Plain", "Hybrid"], widths=0.6,
                     patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#1f6feb", "#e2662f"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(fd, fontsize=10)
axes[0].set_ylabel("Mean absolute error\nover 6 seeds (cycles)")
plt.tight_layout()
plt.savefig("manuscript/icaiet2027/figures/fig_error_dist.pdf", bbox_inches="tight")
plt.savefig("manuscript/icaiet2027/figures/fig_error_dist.png", dpi=150, bbox_inches="tight")
print("Saved fig_error_dist (seed-averaged per-engine error)")
