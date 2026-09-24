"""
Two additional analyses on existing per-engine predictions (no retraining):
 1. Equivalence testing (TOST) of each trend/architecture variant against the plain LSTM.
 2. Seed-budget analysis: how often would a study with k seeds reach the same
    conclusion as the full six-seed study, and how many seeds are needed.
Run from the repository root: python3 analysis/equiv_seedbudget.py
"""
import json, itertools, warnings
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
warnings.filterwarnings("ignore")

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]
DELTA = 1.0            # equivalence margin, cycles of absolute error (fixed before looking at results)
SENS = [0.5, 1.0, 1.5, 2.0]
ALPHA = 0.05

def load(path, key):
    d = json.load(open(path))[key]
    return np.array(d["y_true"]), np.abs(np.array(d["y_pred"]) - np.array(d["y_true"]))

SRC = {  # name -> (path template, key)
    "plain":      ("results/{fd}/seed_{s}/raw_predictions.json", "Plain_LSTM"),
    "noncausal":  ("results/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "causal":     ("results/causal/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "ma":         ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "MA_Hybrid_LSTM"),
    "ema":        ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "EMA_Hybrid_LSTM"),
    "cnn_lstm":   ("results/cnn_lstm/{fd}/seed_{s}/raw_predictions.json", "CNN_LSTM"),
    "gated":      ("results/gated_fusion/{fd}/seed_{s}/raw_predictions.json", "Gated_Fusion"),
}
ERR = {}   # ERR[model][fd] = array (seeds, engines)
for m, (tp, key) in SRC.items():
    ERR[m] = {}
    for fd in DATASETS:
        rows, ytrue0 = [], None
        for s in SEEDS:
            yt, e = load(tp.format(fd=fd, s=s), key)
            if ytrue0 is None: ytrue0 = yt
            assert np.array_equal(yt, ytrue0), f"engine order mismatch {m} {fd} {s}"
            rows.append(e)
        ERR[m][fd] = np.array(rows)

COMPARE = [("noncausal", "plain"), ("causal", "plain"), ("ma", "plain"), ("ema", "plain"),
           ("cnn_lstm", "plain"), ("gated", "plain"), ("ma", "causal")]

# ---------------------------------------------------------------- 1. equivalence
def tost(d, delta):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - stats.t.cdf((m + delta) / se, n - 1)     # H0: diff <= -delta
    p_hi = stats.t.cdf((m - delta) / se, n - 1)          # H0: diff >= +delta
    return max(p_lo, p_hi), m, se, n

out1 = {}
print(f"\n=== 1. EQUIVALENCE (TOST), margin delta = {DELTA} cycles of per-engine absolute error ===")
print("diff = |err|_A - |err|_B on seed-averaged per-engine errors; 90% CI is the TOST-equivalent interval\n")
for A, B in COMPARE:
    rows, pvals = [], []
    for fd in DATASETS:
        d = ERR[A][fd].mean(0) - ERR[B][fd].mean(0)
        p, m, se, n = tost(d, DELTA)
        tcrit = stats.t.ppf(0.95, n - 1)
        lo, hi = m - tcrit * se, m + tcrit * se
        dmin = max(abs(lo), abs(hi))                     # smallest margin at which equivalence holds
        pw = stats.wilcoxon(d).pvalue
        rows.append(dict(fd=fd, n=n, mean_diff=m, ci90=[lo, hi], min_margin=dmin, p_tost=p, p_wilcoxon=pw))
        pvals.append(p)
    padj = multipletests(pvals, method="holm")[1]
    padj_w = multipletests([r["p_wilcoxon"] for r in rows], method="holm")[1]
    for r, pa, pw_ in zip(rows, padj, padj_w):
        r["p_tost_holm"] = pa; r["p_wilcoxon_holm"] = pw_
        equiv = pa < ALPHA
        diff = pw_ < ALPHA
        r["verdict"] = ("equivalent, yet detectably different" if (equiv and diff) else "equivalent" if equiv else "different" if diff else "inconclusive")
    out1[f"{A}_vs_{B}"] = rows
    print(f"{A} vs {B}")
    print(f"  {'':6s}{'mean diff':>10s} {'90% CI':>18s} {'min margin':>11s} {'TOST p(adj)':>12s} {'Wilcoxon p(adj)':>16s}  verdict")
    for r in rows:
        print(f"  {r['fd']:6s}{r['mean_diff']:+10.2f} [{r['ci90'][0]:+6.2f},{r['ci90'][1]:+6.2f}] {r['min_margin']:11.2f} {r['p_tost_holm']:12.4f} {r['p_wilcoxon_holm']:16.4f}  {r['verdict']}")
print("\nSensitivity: number of datasets (of 4) equivalent to plain LSTM at each margin (Holm-adjusted TOST)")
sens = {}
for A, B in COMPARE:
    line = []
    for dl in SENS:
        ps = [tost(ERR[A][fd].mean(0) - ERR[B][fd].mean(0), dl)[0] for fd in DATASETS]
        k = int((multipletests(ps, method="holm")[1] < ALPHA).sum()); line.append(k)
        sens[f"{A}_vs_{B}@{dl}"] = k
    print(f"  {A:>9s} vs {B:6s}: " + "  ".join(f"d={dl}:{k}/4" for dl, k in zip(SENS, line)))

# ---------------------------------------------------------------- 2. seed budget
def flags(errA, errB, subset):
    ps = []
    for fd in DATASETS:
        d = errA[fd][list(subset)].mean(0) - errB[fd][list(subset)].mean(0)
        ps.append(1.0 if not np.any(d) else stats.wilcoxon(d).pvalue)
    raw = tuple(int(p < ALPHA) for p in ps)
    holm = tuple(int(x) for x in multipletests(ps, method="holm")[1] < ALPHA)
    return raw, holm

out2 = {}
print("\n=== 2. SEED BUDGET: agreement of k-seed studies with the six-seed conclusion (all C(6,k) subsets) ===")
for A, B in [("noncausal", "plain"), ("causal", "plain"), ("ma", "causal")]:
    full_raw, full_holm = flags(ERR[A], ERR[B], range(6))
    print(f"\n{A} vs {B}: six-seed raw-significant datasets = {[d for d, f in zip(DATASETS, full_raw) if f]}, Holm-significant = {[d for d, f in zip(DATASETS, full_holm) if f]}")
    print(f"  {'k':>2s} {'subsets':>8s} {'P(any raw sig)':>15s} {'P(raw set = 6-seed set)':>24s} {'P(any Holm sig)':>16s} {'P(Holm set = 6-seed)':>21s}   per-dataset raw-sig rate")
    res = []
    for k in range(1, 7):
        subs = list(itertools.combinations(range(6), k)); F = [flags(ERR[A], ERR[B], s) for s in subs]
        anyraw = np.mean([any(r) for r, h in F]); anyholm = np.mean([any(h) for r, h in F])
        eqraw = np.mean([r == full_raw for r, h in F]); eqholm = np.mean([h == full_holm for r, h in F])
        per = np.mean([r for r, h in F], axis=0)
        res.append(dict(k=k, subsets=len(subs), p_any_raw=anyraw, p_raw_set_equal=eqraw, p_any_holm=anyholm, p_holm_set_equal=eqholm, per_dataset_raw_rate=dict(zip(DATASETS, per.tolist()))))
        print(f"  {k:>2d} {len(subs):>8d} {anyraw:15.2f} {eqraw:24.2f} {anyholm:16.2f} {eqholm:21.2f}   " + " ".join(f"{d[-1]}:{x:.2f}" for d, x in zip(DATASETS, per)))
    out2[f"{A}_vs_{B}"] = dict(full_raw=full_raw, full_holm=full_holm, by_k=res)

print("\n--- Seeds needed for a paired t-test on per-seed dataset RMSE differences (two-sided, 80% power) ---")
def rmse(m, fd):  # per-seed RMSE
    return np.sqrt((ERR[m][fd] ** 2).mean(1))
pow_out = {}
solver = TTestPower()
for A, B in [("noncausal", "plain"), ("causal", "plain"), ("ma", "plain"), ("ma", "causal")]:
    print(f"{A} vs {B}")
    for fd in DATASETS:
        dd = rmse(A, fd) - rmse(B, fd); m, sd = dd.mean(), dd.std(ddof=1)
        def need(alpha, eff):
            try: return float(solver.solve_power(effect_size=eff, alpha=alpha, power=0.8, alternative="two-sided"))
            except Exception: return float("nan")
        n_obs = need(0.05, abs(m) / sd); n_holm = need(0.05 / 4, abs(m) / sd); n_half = need(0.05, 0.5 / sd)
        pow_out[f"{A}_vs_{B}_{fd}"] = dict(mean_rmse_diff=m, sd=sd, seeds_for_observed_effect=n_obs, seeds_holm=n_holm, seeds_for_0p5_cycle=n_half)
        print(f"  {fd}: mean RMSE diff {m:+.2f}, sd across seeds {sd:.2f} -> seeds for observed effect {n_obs:5.1f} (Holm alpha {n_holm:5.1f}); seeds to detect a 0.5-cycle RMSE gap {n_half:5.1f}")

json.dump(dict(delta=DELTA, equivalence=out1, sensitivity=sens, seed_budget=out2, power=pow_out), open("results/equivalence_seed_budget.json", "w"), indent=1, default=float)
print("\nsaved results/equivalence_seed_budget.json")
