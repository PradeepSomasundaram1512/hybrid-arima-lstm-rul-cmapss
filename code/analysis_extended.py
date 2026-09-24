"""
Extended analysis over every available seed (no retraining). For each paired comparison and dataset:
  * per-seed descriptives (RMSE / MAE / PHM, mean +- sd over seeds)
  * seed-level paired tests on per-seed RMSE (Wilcoxon, paired t), Holm across the four datasets
  * two-level (seeds x engines) tests: variance-components t and pigeonhole bootstrap (Owen 2007),
    the larger p kept; Holm across datasets; equivalence margins (90% interval)
  * seed budget: how often a k-seed study reaches the all-seed conclusion (random subsets)
  * seeds needed for 80% power on the per-seed RMSE differences
Run from the repository root: python3 code/analysis_extended.py
"""
import json, os, warnings
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
warnings.filterwarnings("ignore")

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
ALL_SEEDS = [42, 123, 2024, 7, 99, 314] + [s for s in range(1, 16) if s != 7]
B = 40000
DELTA, ALPHA = 1.0, 0.05
rng = np.random.default_rng(20260925)

SRC = {
    "plain":     ("results/{fd}/seed_{s}/raw_predictions.json", "Plain_LSTM"),
    "noncausal": ("results/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "order_sel": ("results/order_selected/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "causal":    ("results/causal/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "ma":        ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "MA_Hybrid_LSTM"),
    "ema":       ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "EMA_Hybrid_LSTM"),
    "pm_plain":  ("results/capacity_controls/{fd}/seed_{s}/raw_predictions.json", "ParamMatched_Plain_LSTM"),
    "dup_plain": ("results/capacity_controls/{fd}/seed_{s}/raw_predictions.json", "DupChannel_Plain_LSTM"),
    "cnn_lstm":  ("results/cnn_lstm/{fd}/seed_{s}/raw_predictions.json", "CNN_LSTM"),
    "gated":     ("results/gated_fusion/{fd}/seed_{s}/raw_predictions.json", "Gated_Fusion"),
}
COMPARE = [("noncausal", "plain"), ("causal", "plain"), ("order_sel", "plain"), ("order_sel", "noncausal"),
           ("ma", "plain"), ("ema", "plain"), ("ma", "causal"), ("cnn_lstm", "plain"), ("gated", "plain"),
           ("noncausal", "pm_plain"), ("causal", "pm_plain"), ("noncausal", "dup_plain"), ("pm_plain", "plain"), ("dup_plain", "plain"),
           ("causal", "noncausal"), ("ma", "order_sel"), ("ma", "noncausal"), ("order_sel", "pm_plain"), ("order_sel", "dup_plain"),
           ("ma", "pm_plain"), ("ma", "dup_plain"), ("causal", "dup_plain")]


def phm(d):
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


# DATA[m][fd] = dict(seeds=[...], abs=(S,E), rmse=(S,), mae=(S,), phm=(S,))
DATA = {}
for m, (tp, key) in SRC.items():
    DATA[m] = {}
    for fd in DATASETS:
        seeds, A, R, M, P = [], [], [], [], []
        y0 = None
        for s in ALL_SEEDS:
            p = tp.format(fd=fd, s=s)
            if not os.path.exists(p): continue
            d = json.load(open(p))[key]
            yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
            y0 = yt if y0 is None else y0
            assert np.array_equal(yt, y0), (m, fd, s)
            e = yp - yt
            seeds.append(s); A.append(np.abs(e)); R.append(float(np.sqrt((e ** 2).mean()))); M.append(float(np.abs(e).mean())); P.append(phm(e))
        if seeds:
            DATA[m][fd] = dict(seeds=seeds, abs=np.array(A), rmse=np.array(R), mae=np.array(M), phm=np.array(P))


def paired(A, Bm, fd):
    a, b = DATA[A].get(fd), DATA[Bm].get(fd)
    if a is None or b is None: return None
    common = [s for s in a["seeds"] if s in b["seeds"]]
    if len(common) < 3: return None
    ia = [a["seeds"].index(s) for s in common]; ib = [b["seeds"].index(s) for s in common]
    return dict(seeds=common, D=a["abs"][ia] - b["abs"][ib], dr=a["rmse"][ia] - b["rmse"][ib])


def varcomp(D):
    S, E = D.shape; m = D.mean(); r, c = D.mean(1), D.mean(0)
    ms_s = E * ((r - m) ** 2).sum() / (S - 1); ms_e = S * ((c - m) ** 2).sum() / (E - 1)
    ms_r = ((D - r[:, None] - c[None, :] + m) ** 2).sum() / ((S - 1) * (E - 1))
    if max(0.0, (ms_s - ms_r) / E) > 0:
        a = np.array([ms_s, ms_e, -ms_r]) / (S * E); dfs = np.array([S - 1, E - 1, (S - 1) * (E - 1)])
        var = a.sum(); df = var ** 2 / ((a ** 2) / dfs).sum()
    else:
        var = ms_e / (S * E); df = E - 1
    return m, np.sqrt(max(var, 1e-12)), max(df, 1.0)


def boot(D):
    S, E = D.shape; out = np.empty(B)
    for b in range(B):
        out[b] = D[rng.integers(0, S, S)].mean(0)[rng.integers(0, E, E)].mean()
    return out


def p_two(m, se, df): return 2 * (1 - stats.t.cdf(abs(m) / se, df))


OUT = dict(seeds_available={}, descriptives={}, comparisons={}, seed_budget={}, power={})
print("=== descriptives: mean +- sd over available seeds (RMSE) ===")
for m in SRC:
    row = {}
    for fd in DATASETS:
        d = DATA[m].get(fd)
        if d: row[fd] = dict(n_seeds=len(d["seeds"]), rmse_mean=float(d["rmse"].mean()), rmse_sd=float(d["rmse"].std(ddof=1)),
                             mae_mean=float(d["mae"].mean()), mae_sd=float(d["mae"].std(ddof=1)),
                             phm_mean=float(d["phm"].mean()), phm_sd=float(d["phm"].std(ddof=1)))
    if row:
        OUT["descriptives"][m] = row
        print(f"{m:10s} " + "  ".join(f"{fd}:{v['rmse_mean']:.2f}+-{v['rmse_sd']:.2f}(n={v['n_seeds']})" for fd, v in row.items()))

print("\n=== paired comparisons (A minus B; negative favours A). p: seed-level Wilcoxon | two-level (larger of variance-components, bootstrap), Holm across 4 datasets ===")
for A, Bm in COMPARE:
    rows = []
    for fd in DATASETS:
        pr = paired(A, Bm, fd)
        if pr is None: continue
        D, dr = pr["D"], pr["dr"]; S = D.shape[0]
        m, se, df = varcomp(D); bb = boot(D)
        tc = stats.t.ppf(0.95, df)
        ci90 = (min(m - tc * se, np.percentile(bb, 5)), max(m + tc * se, np.percentile(bb, 95)))
        p_vc = p_two(m, se, df); p_b = min(1.0, 2 * (min((bb <= 0).sum(), (bb >= 0).sum()) + 1) / (B + 1))
        plo, phi = ((bb <= -DELTA).sum() + 1) / (B + 1), ((bb >= DELTA).sum() + 1) / (B + 1)
        p_tost = max(plo, phi, 1 - stats.t.cdf((m + DELTA) / se, df), stats.t.cdf((m - DELTA) / se, df))
        try: pw = float(stats.wilcoxon(dr).pvalue)
        except Exception: pw = 1.0
        rows.append(dict(fd=fd, n_seeds=S, n_engines=D.shape[1], mean_abs_err_diff=float(m), mean_rmse_diff=float(dr.mean()), sd_rmse_diff=float(dr.std(ddof=1)),
                         ci90=[float(ci90[0]), float(ci90[1])], min_margin=float(max(abs(ci90[0]), abs(ci90[1]))),
                         p_seed_wilcoxon=pw, p_seed_t=float(stats.ttest_1samp(dr, 0).pvalue),
                         p_twolevel_varcomp=float(p_vc), p_twolevel_boot=float(p_b), p_twolevel=float(max(p_vc, p_b)), p_tost=float(p_tost)))
    if not rows: continue
    for key in ("p_seed_wilcoxon", "p_seed_t", "p_twolevel", "p_tost"):
        adj = multipletests([r[key] for r in rows], method="holm")[1]
        for r, a in zip(rows, adj): r[key + "_holm"] = float(a)
    for r in rows:
        eq, df_ = r["p_tost_holm"] < ALPHA, r["p_twolevel_holm"] < ALPHA
        r["verdict"] = "equivalent, yet different" if eq and df_ else "equivalent" if eq else "different" if df_ else "inconclusive"
    OUT["comparisons"][f"{A}_vs_{Bm}"] = rows
    print(f"\n{A} vs {Bm}")
    print(f"  {'':6s}{'n':>3s} {'dMAE':>6s} {'dRMSE':>6s} {'sd':>5s} {'seedWilx(H)':>12s} {'2-level(H)':>11s} {'min margin':>11s}  verdict")
    for r in rows:
        print(f"  {r['fd']:6s}{r['n_seeds']:>3d} {r['mean_abs_err_diff']:+6.2f} {r['mean_rmse_diff']:+6.2f} {r['sd_rmse_diff']:5.2f} {r['p_seed_wilcoxon_holm']:12.4f} {r['p_twolevel_holm']:11.4f} {r['min_margin']:11.2f}  {r['verdict']}")

print("\n=== seed budget: random k-seed subsets vs the all-seed two-level conclusion (variance-components test, Holm) ===")
def flags(A, Bm, pick):
    ps = []
    for fd in DATASETS:
        pr = paired(A, Bm, fd)
        D = pr["D"][pick]; m, se, df = varcomp(D); ps.append(p_two(m, se, df))
    return tuple(int(x) for x in multipletests(ps, method="holm")[1] < ALPHA)
for A, Bm in [("noncausal", "plain"), ("causal", "plain"), ("ma", "causal"), ("order_sel", "plain")]:
    prs = [paired(A, Bm, fd) for fd in DATASETS]
    if any(p is None for p in prs): continue
    S = min(p["D"].shape[0] for p in prs)
    if S < 8: continue
    full = flags(A, Bm, list(range(S))); res = []
    print(f"{A} vs {Bm}: {S} seeds, all-seed Holm-significant datasets = {[d for d, f in zip(DATASETS, full) if f]}")
    for k in sorted({2, 3, 4, 6, 8, 10, 15} & set(range(2, S))):
        F = [flags(A, Bm, list(rng.choice(S, k, replace=False))) for _ in range(300)]
        res.append(dict(k=k, p_any_holm_sig=float(np.mean([any(f) for f in F])), p_same_set_as_all=float(np.mean([f == full for f in F]))))
        print(f"   k={k:>2d}: P(any Holm-significant)={res[-1]['p_any_holm_sig']:.2f}  P(same set as all-seed)={res[-1]['p_same_set_as_all']:.2f}")
    OUT["seed_budget"][f"{A}_vs_{Bm}"] = dict(n_seeds=S, all_seed_set=list(full), by_k=res)


print("\n=== sign stability: does a k-seed study get the SIGN of the mean RMSE difference right? (reference = all available seeds) ===")
ORIG6 = [42, 123, 2024, 7, 99, 314]
OUT["sign_stability"] = {}
for A, Bm in [("noncausal", "plain"), ("causal", "plain"), ("order_sel", "plain"), ("ma", "plain"), ("ma", "causal")]:
    line = []
    for fd in DATASETS:
        pr = paired(A, Bm, fd)
        if pr is None or len(pr["seeds"]) < 15: continue
        dr, seeds = pr["dr"], pr["seeds"]; ref = np.sign(dr.mean())
        o6 = dr[[seeds.index(x) for x in ORIG6]].mean()
        rec = dict(fd=fd, mean_20=float(dr.mean()), mean_orig6=float(o6), sign_flips_in_orig6=bool(np.sign(o6) != ref))
        for k in (3, 6, 10):
            rec[f"p_sign_ok_k{k}"] = float(np.mean([np.sign(dr[rng.choice(len(dr), k, replace=False)].mean()) == ref for _ in range(2000)]))
        line.append(rec)
    OUT["sign_stability"][f"{A}_vs_{Bm}"] = line
    print(f"{A} vs {Bm}")
    for r in line:
        print(f"  {r['fd']}: 20-seed dRMSE {r['mean_20']:+.2f} | original six seeds {r['mean_orig6']:+.2f}{'  <-- SIGN FLIPS' if r['sign_flips_in_orig6'] else ''} | P(sign right) k=3 {r['p_sign_ok_k3']:.2f}, k=6 {r['p_sign_ok_k6']:.2f}, k=10 {r['p_sign_ok_k10']:.2f}")

print("\n=== seeds needed for 80% power (paired t on per-seed RMSE differences; alpha 0.05 and Holm 0.0125) ===")
solver = TTestPower()
for A, Bm in [("noncausal", "plain"), ("causal", "plain"), ("ma", "causal"), ("order_sel", "plain")]:
    for fd in DATASETS:
        pr = paired(A, Bm, fd)
        if pr is None: continue
        dr = pr["dr"]; m, sd = dr.mean(), dr.std(ddof=1)
        def need(alpha, eff):
            try: return float(solver.solve_power(effect_size=eff, alpha=alpha, power=0.8, alternative="two-sided"))
            except Exception: return float("nan")
        OUT["power"][f"{A}_vs_{Bm}_{fd}"] = dict(n_seeds=len(dr), mean=float(m), sd=float(sd), seeds_observed=need(0.05, abs(m) / sd), seeds_observed_holm=need(0.0125, abs(m) / sd), seeds_half_cycle=need(0.05, 0.5 / sd))
        print(f"  {A} vs {Bm} {fd}: mean {m:+.2f}, sd {sd:.2f} -> {OUT['power'][f'{A}_vs_{Bm}_{fd}']['seeds_observed']:.0f} seeds (Holm {OUT['power'][f'{A}_vs_{Bm}_{fd}']['seeds_observed_holm']:.0f}); half-cycle gap {OUT['power'][f'{A}_vs_{Bm}_{fd}']['seeds_half_cycle']:.0f}")

OUT["seeds_available"] = {m: {fd: len(DATA[m][fd]["seeds"]) for fd in DATA[m]} for m in DATA}
json.dump(OUT, open("results/extended_analysis.json", "w"), indent=1, default=float)
print("\nsaved results/extended_analysis.json")
