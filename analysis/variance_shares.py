"""Share of the variance of each paired mean difference that comes from training seeds vs test-engine sampling.
Var(mean) = sigma2_seed/S + [sigma2_engine/E + sigma2_resid/(S*E)]; the seed component is floored at zero.
Run from the repository root: python3 analysis/variance_shares.py"""
import json, os, numpy as np
DS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314] + [s for s in range(1, 16) if s != 7]
SRC = {"plain": ("results/{fd}/seed_{s}/raw_predictions.json", "Plain_LSTM"), "noncausal": ("results/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
       "order_sel": ("results/order_selected/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"), "causal": ("results/causal/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
       "ma": ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "MA_Hybrid_LSTM"), "ema": ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "EMA_Hybrid_LSTM"),
       "pm_plain": ("results/capacity_controls/{fd}/seed_{s}/raw_predictions.json", "ParamMatched_Plain_LSTM"), "dup_plain": ("results/capacity_controls/{fd}/seed_{s}/raw_predictions.json", "DupChannel_Plain_LSTM")}
def load(m, fd):
    tp, k = SRC[m]; out = []
    for s in SEEDS:
        d = json.load(open(tp.format(fd=fd, s=s)))[k]; out.append(np.abs(np.array(d["y_pred"]) - np.array(d["y_true"])))
    return np.array(out)
# exactly the 12 comparisons of Table II
PAIRS = [("noncausal", "plain"), ("order_sel", "plain"), ("causal", "plain"), ("ma", "plain"), ("ema", "plain"), ("pm_plain", "plain"), ("dup_plain", "plain"),
         ("order_sel", "noncausal"), ("causal", "noncausal"), ("ma", "causal"), ("noncausal", "dup_plain"), ("ma", "dup_plain")]
res = {}; shares = []
for fd in DS:
    E = {m: load(m, fd) for m in SRC}
    for a, b in PAIRS:
        D = E[a] - E[b]; S, n = D.shape; m = D.mean(); r, c = D.mean(1), D.mean(0)
        ms_s = n * ((r - m) ** 2).sum() / (S - 1); ms_e = S * ((c - m) ** 2).sum() / (n - 1)
        ms_r = ((D - r[:, None] - c[None, :] + m) ** 2).sum() / ((S - 1) * (n - 1))
        seed_part = max(0.0, (ms_s - ms_r) / n) / S; eng_part = ms_e / (S * n)
        sh = seed_part / (seed_part + eng_part); shares.append(sh); res[f"{a}_vs_{b}_{fd}"] = dict(seed_share=float(sh), se_seed_only=float(np.sqrt(seed_part)), se_engine_part=float(np.sqrt(eng_part)))
q = np.percentile(shares, [10, 25, 50, 75, 90])
print(f"{len(shares)} cells; seed share of variance of the mean difference: median {q[2]:.2f}, IQR [{q[1]:.2f}, {q[3]:.2f}], 10-90% [{q[0]:.2f}, {q[4]:.2f}]")
se_seed = np.median([v["se_seed_only"] for v in res.values()]); se_eng = np.median([v["se_engine_part"] for v in res.values()])
print(f"median standard error from seeds alone {se_seed:.2f} cycles vs from engine sampling {se_eng:.2f} cycles")
json.dump(dict(cells=res, quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"], map(float, q)))), open("results/variance_shares.json", "w"), indent=1)
