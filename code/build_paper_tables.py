"""Fill %%MEANS%% and %%PAIRED%% in the template from results/extended_analysis.json (no hand-typed numbers)."""
import json, sys
R = "/Users/pradeepsomasundaram/Library/Mobile Documents/com~apple~CloudDocs/Documents/IEEE_BigData_2026_ARIMA_LSTM_Paper/results/extended_analysis.json"
J = json.load(open(R)); DS = ["FD001", "FD002", "FD003", "FD004"]

names = [("plain", "Plain LSTM"), ("noncausal", "Fixed-order ARIMA"), ("order_sel", "AIC-selected ARIMA"), ("causal", "Causal ARIMA"),
         ("ma", "Moving average"), ("ema", "Exp.\\ smoothing"), ("pm_plain", "Param-matched plain"), ("dup_plain", "Dup.-input plain"), ("cnn_lstm", "CNN-LSTM")]
D = J["descriptives"]
best = {fd: min(D[m][fd]["rmse_mean"] for m, _ in names if m in D and D[m][fd]["n_seeds"] == 20) for fd in DS}
rows = []
for m, label in names:
    if m not in D: continue
    cells = []
    for fd in DS:
        v = D[m][fd]; s = f"{v['rmse_mean']:.2f}{{\\tiny$\\pm${v['rmse_sd']:.2f}}}"
        cells.append(f"\\textbf{{{s}}}" if (v["n_seeds"] == 20 and abs(v["rmse_mean"] - best[fd]) < 1e-9) else s)
    n = D[m]["FD001"]["n_seeds"]
    rows.append(f"{label}{'' if n == 20 else f' ({n} seeds)'} & " + " & ".join(cells) + " \\\\")
means = "\n".join(rows)

pairs = [("noncausal_vs_plain", "Fixed ARIMA $-$ plain"), ("order_sel_vs_plain", "AIC ARIMA $-$ plain"), ("causal_vs_plain", "Causal ARIMA $-$ plain"),
         ("ma_vs_plain", "Moving avg $-$ plain"), ("ema_vs_plain", "Exp.\\ smooth $-$ plain"), ("pm_plain_vs_plain", "Param-matched $-$ plain"),
         ("dup_plain_vs_plain", "Dup.\\ input $-$ plain"), ("order_sel_vs_noncausal", "AIC $-$ fixed ARIMA"), ("causal_vs_noncausal", "Causal $-$ fixed ARIMA"),
         ("ma_vs_causal", "Moving avg $-$ causal"), ("noncausal_vs_dup_plain", "Fixed ARIMA $-$ dup.\\ input"), ("ma_vs_dup_plain", "Moving avg $-$ dup.\\ input")]
rows = []; n_seed = n_two = 0
for key, label in pairs:
    if key not in J["comparisons"]: continue
    cells = []
    for r in J["comparisons"][key]:
        mark = ""
        if r["p_seed_wilcoxon_holm"] < 0.05: mark += "$^*$"; n_seed += 1
        if r["p_twolevel_holm"] < 0.05: mark += "$^\\dagger$"; n_two += 1
        cells.append(f"{r['mean_rmse_diff']:+.2f}{mark}".replace("-", "$-$"))
    rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
paired = "\n".join(rows)
open("means.tex", "w").write(means); open("paired.tex", "w").write(paired)
print("cells significant: seed-level", n_seed, "| two-level", n_two, "of", 4 * len(rows))

# ---- equivalence-margin table
import numpy as np
eq_pairs = [("noncausal_vs_plain","Fixed ARIMA $-$ plain"),("order_sel_vs_plain","AIC ARIMA $-$ plain"),("causal_vs_plain","Causal ARIMA $-$ plain"),
            ("ma_vs_plain","Moving avg $-$ plain"),("ema_vs_plain","Exp.\\ smooth $-$ plain"),("pm_plain_vs_plain","Param-matched $-$ plain"),
            ("dup_plain_vs_plain","Dup.\\ input $-$ plain"),("causal_vs_noncausal","Causal $-$ fixed ARIMA"),("ma_vs_causal","Moving avg $-$ causal")]
rows = []
for key, label in eq_pairs:
    cells = []
    for r in J["comparisons"][key]:
        v = f"{r['min_margin']:.2f}"; cells.append(f"\\textbf{{{v}}}" if r["min_margin"] <= 1.0 else v)
    rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
open("equiv.tex", "w").write("\n".join(rows))
# ---- seed budget table
S = J["sign_stability"]; ks = (3, 6, 10)
cells = [c for v in S.values() for c in v]
med = [np.median([c[f"p_sign_ok_k{k}"] for c in cells]) for k in ks]; mn = [min(c[f"p_sign_ok_k{k}"] for c in cells) for k in ks]
sb = J["seed_budget"]; anysig = []
for k in ks:
    vals = [next(x["p_any_holm_sig"] for x in v["by_k"] if x["k"] == k) for v in sb.values() if any(x["k"] == k for x in v["by_k"])]
    anysig.append(np.mean(vals))
rows = ["Sign recovered, median over %d cells & " % len(cells) + " & ".join(f"{x:.2f}" for x in med) + " \\\\",
        "Sign recovered, worst cell & " + " & ".join(f"{x:.2f}" for x in mn) + " \\\\",
        "Any two-level significant cell (mean) & " + " & ".join(f"{x:.2f}" for x in anysig) + " \\\\"]
open("budget.tex", "w").write("\n".join(rows))
print("equiv and budget tables written; flips in original six seeds:", sum(c["sign_flips_in_orig6"] for c in cells), "of", len(cells))

# ---- MAE / PHM table
mm = [("plain","Plain LSTM"),("noncausal","Fixed ARIMA"),("order_sel","AIC ARIMA"),("causal","Causal ARIMA"),("ma","Moving avg")]
rows = []
for m, label in mm:
    d = D[m]
    mae = " & ".join(f"{d[fd]['mae_mean']:.2f}{{\\tiny$\\pm${d[fd]['mae_sd']:.2f}}}" for fd in DS)
    phm = " & ".join(f"{d[fd]['phm_mean']:.0f}" for fd in DS)
    rows.append(f"{label} & {mae} & {phm} \\\\")
open("metrics.tex", "w").write("\n".join(rows))

# ---- full p-value table (seed-level | two-level, Holm)
def fp(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for key, label in pairs:
    if key not in J["comparisons"]: continue
    cells = []
    for r in J["comparisons"][key]:
        a, c = r["p_seed_wilcoxon_holm"], r["p_twolevel_holm"]
        sa = f"\\textbf{{{fp(a)}}}" if a < 0.05 else fp(a); sc = f"\\textbf{{{fp(c)}}}" if c < 0.05 else fp(c)
        cells.append(f"{sa} / {sc}")
    rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
open("pvals.tex", "w").write("\n".join(rows))
