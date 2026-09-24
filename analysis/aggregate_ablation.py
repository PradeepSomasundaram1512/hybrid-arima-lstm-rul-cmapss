import json
import numpy as np

SEEDS = [42, 123, 2024]
WINDOWS = [10, 20, 30, 50]

data = {seed: json.load(open(f"results/ablation_window/seed_{seed}/ablation_results.json")) for seed in SEEDS}

agg = {}
for W in WINDOWS:
    plain_vals = np.array([data[s][str(W)]["plain"]["RMSE"] for s in SEEDS])
    hybrid_vals = np.array([data[s][str(W)]["hybrid"]["RMSE"] for s in SEEDS])
    agg[W] = {
        "plain_mean": float(plain_vals.mean()), "plain_std": float(plain_vals.std(ddof=1)),
        "hybrid_mean": float(hybrid_vals.mean()), "hybrid_std": float(hybrid_vals.std(ddof=1)),
        "plain_vals": plain_vals.tolist(), "hybrid_vals": hybrid_vals.tolist(),
    }
    print(f"W={W}: plain={agg[W]['plain_mean']:.2f}+/-{agg[W]['plain_std']:.2f} "
          f"hybrid={agg[W]['hybrid_mean']:.2f}+/-{agg[W]['hybrid_std']:.2f} "
          f"delta={agg[W]['plain_mean']-agg[W]['hybrid_mean']:+.2f}")

with open("results/ablation_window/aggregated_3seed.json", "w") as f:
    json.dump(agg, f, indent=2)

print("\n=== LaTeX table ===")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Window-length ablation on FD001, mean $\pm$ std over 3 seeds (test RMSE, cycles).}")
print(r"\label{tab:ablation}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"$W$ (cycles) & " + " & ".join(str(w) for w in WINDOWS) + r" \\")
print(r"\midrule")
print("Plain LSTM & " + " & ".join(f"{agg[w]['plain_mean']:.2f} $\\pm$ {agg[w]['plain_std']:.2f}" for w in WINDOWS) + r" \\")
print("Hybrid ARIMA-LSTM & " + " & ".join(f"{agg[w]['hybrid_mean']:.2f} $\\pm$ {agg[w]['hybrid_std']:.2f}" for w in WINDOWS) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
