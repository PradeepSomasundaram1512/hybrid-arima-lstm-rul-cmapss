import json
import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

agg = {}
for fd in DATASETS:
    fixed_rmse, sel_rmse, plain_rmse = [], [], []
    for seed in SEEDS:
        d_sel = json.load(open(f"results/order_selected/{fd}/seed_{seed}/summary_metrics.json"))
        d_fix = json.load(open(f"results/{fd}/seed_{seed}/summary_metrics.json"))
        sel_rmse.append(d_sel["Hybrid_ARIMA_LSTM"]["RMSE"])
        fixed_rmse.append(d_fix["Hybrid_ARIMA_LSTM"]["RMSE"])
        plain_rmse.append(d_fix["Plain_LSTM"]["RMSE"])
    fixed_rmse, sel_rmse, plain_rmse = map(np.array, (fixed_rmse, sel_rmse, plain_rmse))
    stat, p = wilcoxon(sel_rmse, fixed_rmse)
    n_sel_better = int((sel_rmse < fixed_rmse).sum())
    stat2, p_vs_plain = wilcoxon(sel_rmse, plain_rmse)
    n_sel_beats_plain = int((sel_rmse < plain_rmse).sum())
    agg[fd] = {
        "plain_mean": float(plain_rmse.mean()), "plain_std": float(plain_rmse.std(ddof=1)),
        "fixed_mean": float(fixed_rmse.mean()), "fixed_std": float(fixed_rmse.std(ddof=1)),
        "sel_mean": float(sel_rmse.mean()), "sel_std": float(sel_rmse.std(ddof=1)),
        "wilcoxon_p_vs_fixed": float(p), "n_sel_better_vs_fixed": n_sel_better,
        "wilcoxon_p_vs_plain": float(p_vs_plain), "n_sel_better_vs_plain": n_sel_beats_plain,
        "n_seeds": len(SEEDS),
    }
    print(f"{fd}: plain={agg[fd]['plain_mean']:.2f}+/-{agg[fd]['plain_std']:.2f}  "
          f"fixed_hybrid={agg[fd]['fixed_mean']:.2f}+/-{agg[fd]['fixed_std']:.2f}  "
          f"selected_hybrid={agg[fd]['sel_mean']:.2f}+/-{agg[fd]['sel_std']:.2f}  "
          f"[selected vs fixed: better on {n_sel_better}/6, p={p:.3f}] "
          f"[selected vs plain: better on {n_sel_beats_plain}/6, p={p_vs_plain:.3f}]")

with open("results/order_selected/aggregated_6seed.json", "w") as f:
    json.dump(agg, f, indent=2)

print("\n=== LaTeX table ===")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Hybrid model RMSE with AIC-selected vs.\ fixed ARIMA order, mean $\pm$ std over 6 seeds. Plain LSTM shown for reference.}")
print(r"\label{tab:orderselect}")
print(r"\footnotesize")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Dataset & Plain & Fixed order & Selected order \\")
print(r"\midrule")
for fd in DATASETS:
    a = agg[fd]
    print(f"{fd} & {a['plain_mean']:.2f}{{\\tiny$\\pm${a['plain_std']:.2f}}} & "
          f"{a['fixed_mean']:.2f}{{\\tiny$\\pm${a['fixed_std']:.2f}}} & "
          f"{a['sel_mean']:.2f}{{\\tiny$\\pm${a['sel_std']:.2f}}} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
