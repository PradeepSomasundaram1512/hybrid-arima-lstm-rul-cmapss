import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

agg = json.load(open("results/ablation_window/aggregated_3seed.json"))
WINDOWS = [10, 20, 30, 50]

fig, ax = plt.subplots(figsize=(5.4, 3.3))
plain_means = [agg[str(w)]["plain_mean"] for w in WINDOWS]
plain_stds = [agg[str(w)]["plain_std"] for w in WINDOWS]
hybrid_means = [agg[str(w)]["hybrid_mean"] for w in WINDOWS]
hybrid_stds = [agg[str(w)]["hybrid_std"] for w in WINDOWS]

ax.errorbar(WINDOWS, plain_means, yerr=plain_stds, marker="o", label="Plain LSTM", color="#1f6feb", capsize=3)
ax.errorbar(WINDOWS, hybrid_means, yerr=hybrid_stds, marker="o", label="Hybrid ARIMA-LSTM", color="#e2662f", capsize=3)
ax.set_xlabel("Window length $W$ (cycles)")
ax.set_ylabel("Test RMSE (cycles)")
ax.set_title("FD001: window length vs. RMSE, mean ± std over 3 seeds")
ax.legend(fontsize=8)
ax.set_xticks(WINDOWS)
plt.tight_layout()
plt.savefig("paper/fig_ablation.pdf", bbox_inches="tight")
plt.savefig("paper/fig_ablation.png", dpi=150, bbox_inches="tight")
print("Saved fig_ablation (3-seed averaged)")
