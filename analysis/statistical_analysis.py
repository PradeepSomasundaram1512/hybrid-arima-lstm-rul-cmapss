import json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

datasets = ["FD001", "FD002", "FD004"]
preds = {fd: json.load(open(f"results/{fd}/raw_predictions.json")) for fd in datasets}

print("=== Paired Wilcoxon signed-rank test: |error_hybrid| vs |error_plain| per engine ===")
stats_out = {}
for fd in datasets:
    plain = preds[fd]["Plain_LSTM"]
    hybrid = preds[fd]["Hybrid_ARIMA_LSTM"]
    yt = np.array(plain["y_true"])
    err_plain = np.abs(np.array(plain["y_pred"]) - yt)
    err_hybrid = np.abs(np.array(hybrid["y_pred"]) - yt)
    diff = err_hybrid - err_plain  # negative => hybrid better on that engine
    stat, p = wilcoxon(err_hybrid, err_plain)
    n_hybrid_better = int((diff < 0).sum())
    n_plain_better = int((diff > 0).sum())
    n_tied = int((diff == 0).sum())
    median_diff = float(np.median(diff))
    stats_out[fd] = {
        "wilcoxon_stat": float(stat), "p_value": float(p),
        "n_hybrid_better": n_hybrid_better, "n_plain_better": n_plain_better, "n_tied": n_tied,
        "median_diff_abs_error": median_diff, "n_total": len(yt),
    }
    print(f"{fd}: W={stat:.1f} p={p:.4f} | hybrid better on {n_hybrid_better}/{len(yt)} engines "
          f"| median(|err_hybrid|-|err_plain|)={median_diff:.2f}")

with open("results/wilcoxon_tests.json", "w") as f:
    json.dump(stats_out, f, indent=2)

# ---- Error distribution boxplot across datasets, hybrid vs plain ----
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
for ax, fd in zip(axes, datasets):
    plain = preds[fd]["Plain_LSTM"]
    hybrid = preds[fd]["Hybrid_ARIMA_LSTM"]
    yt = np.array(plain["y_true"])
    err_plain = np.abs(np.array(plain["y_pred"]) - yt)
    err_hybrid = np.abs(np.array(hybrid["y_pred"]) - yt)
    bp = ax.boxplot([err_plain, err_hybrid], tick_labels=["Plain", "Hybrid"], widths=0.6,
                     patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#1f6feb", "#e2662f"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(fd, fontsize=10)
axes[0].set_ylabel("Absolute error (cycles)")
plt.tight_layout()
plt.savefig("manuscript/icaiet2027/figures/fig_error_dist.pdf", bbox_inches="tight")
plt.savefig("manuscript/icaiet2027/figures/fig_error_dist.png", dpi=150, bbox_inches="tight")
print("Saved paper/fig_error_dist.{pdf,png}")
