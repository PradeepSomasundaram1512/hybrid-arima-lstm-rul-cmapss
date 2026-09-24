"""Figures for the 20-seed study. Run from the repository root."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314] + [s for s in range(1, 16) if s != 7]
SRC = {"Plain": ("results/{fd}/seed_{s}/raw_predictions.json", "Plain_LSTM"),
       "Fixed ARIMA": ("results/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
       "AIC ARIMA": ("results/order_selected/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
       "Causal ARIMA": ("results/causal/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
       "Moving avg": ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "MA_Hybrid_LSTM"),
       "Exp. smooth": ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "EMA_Hybrid_LSTM"),
       "Dup. channel": ("results/capacity_controls/{fd}/seed_{s}/raw_predictions.json", "DupChannel_Plain_LSTM")}

def rmse_table():
    R = {}
    for m, (tp, k) in SRC.items():
        for fd in DS:
            v = []
            for s in SEEDS:
                d = json.load(open(tp.format(fd=fd, s=s)))[k]
                v.append(np.sqrt(np.mean((np.array(d["y_pred"]) - np.array(d["y_true"])) ** 2)))
            R[(m, fd)] = np.array(v)
    return R

R = rmse_table()
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
os.makedirs("manuscript/icaiet2027/figures", exist_ok=True)
cols = {"Plain": "#444444", "Fixed ARIMA": "#c0504d", "AIC ARIMA": "#8064a2", "Causal ARIMA": "#e69138", "Moving avg": "#4f81bd", "Exp. smooth": "#76a5af", "Dup. channel": "#999999"}

# Fig A: paired per-seed RMSE differences vs plain
variants = ["Fixed ARIMA", "AIC ARIMA", "Causal ARIMA", "Moving avg", "Exp. smooth", "Dup. channel"]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
rng = np.random.default_rng(0)
for ax, fd in zip(axes, DS):
    for i, v in enumerate(variants):
        d = R[(v, fd)] - R[("Plain", fd)]
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=5, color=cols[v], alpha=0.55, linewidths=0)
        ax.hlines(d.mean(), i - 0.3, i + 0.3, color="black", lw=1.4)
    ax.axhline(0, color="gray", lw=0.6, ls="--"); ax.set_title(fd)
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=60, ha="right")
axes[0].set_ylabel("RMSE minus plain (cycles)")
plt.tight_layout(); plt.savefig("manuscript/icaiet2027/figures/fig_paired20.pdf"); plt.savefig("manuscript/icaiet2027/figures/fig_paired20.png", dpi=200); plt.close()

# Fig B: cumulative mean of (fixed ARIMA - plain) as seeds are added in the original order
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
for ax, (v, title) in zip(axes, [("Fixed ARIMA", "Fixed-order ARIMA hybrid minus plain"), ("AIC ARIMA", "AIC-selected ARIMA hybrid minus plain")]):
    for fd, c in zip(DS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]):
        d = R[(v, fd)] - R[("Plain", fd)]
        ax.plot(range(1, len(d) + 1), np.cumsum(d) / np.arange(1, len(d) + 1), color=c, lw=1.2, marker="o", ms=2, label=fd)
    ax.axhline(0, color="gray", lw=0.6, ls="--"); ax.axvspan(0.5, 3.5, color="#eeeeee", zorder=0); ax.axvline(6.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("number of seeds averaged"); ax.set_title(title); ax.set_xticks([1, 3, 6, 10, 15, 20])
axes[0].set_ylabel("mean RMSE difference (cycles)"); axes[0].legend(ncol=2, frameon=False)
plt.tight_layout(); plt.savefig("manuscript/icaiet2027/figures/fig_cumulative.pdf"); plt.savefig("manuscript/icaiet2027/figures/fig_cumulative.png", dpi=200); plt.close()

# Fig C: means +- sd over 20 seeds
models = ["Plain", "Fixed ARIMA", "AIC ARIMA", "Causal ARIMA", "Moving avg"]
fig, ax = plt.subplots(figsize=(3.4, 2.4)); w = 0.16
for i, m in enumerate(models):
    mu = [R[(m, fd)].mean() for fd in DS]; sd = [R[(m, fd)].std(ddof=1) for fd in DS]
    ax.bar(np.arange(4) + (i - 2) * w, mu, w, yerr=sd, color=cols[m], capsize=1.5, error_kw={"lw": 0.7}, label=m)
ax.set_ylim(12, 17); ax.set_xticks(range(4)); ax.set_xticklabels(DS); ax.set_ylabel("test RMSE (cycles)")
ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("manuscript/icaiet2027/figures/fig_means20.pdf"); plt.savefig("manuscript/icaiet2027/figures/fig_means20.png", dpi=200); plt.close()
print("saved figures: fig_paired20, fig_cumulative, fig_means20")
