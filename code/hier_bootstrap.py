"""
Two-level (seeds x engines) uncertainty for paired model comparisons, no retraining.

Seeds and test engines form a CROSSED design (every engine is scored under every seed), so
we use the pigeonhole bootstrap (Owen 2007): resample seeds and engines independently.
Because only six seeds exist, a percentile bootstrap can be too narrow; we therefore also
report a variance-components (ANOVA) t-interval with Satterthwaite degrees of freedom and
use the more conservative of the two for equivalence claims.

Statistic: mean over seeds and engines of |err|_A - |err|_B.
Run from the repository root: python3 code/hier_bootstrap.py
"""
import json, itertools, warnings
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]
B = 10000
DELTA, ALPHA = 1.0, 0.05
rng = np.random.default_rng(20260924)

SRC = {
    "plain":     ("results/{fd}/seed_{s}/raw_predictions.json", "Plain_LSTM"),
    "noncausal": ("results/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "causal":    ("results/causal/{fd}/seed_{s}/raw_predictions.json", "Hybrid_ARIMA_LSTM"),
    "ma":        ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "MA_Hybrid_LSTM"),
    "ema":       ("results/trend_controls/{fd}/seed_{s}/raw_predictions.json", "EMA_Hybrid_LSTM"),
    "cnn_lstm":  ("results/cnn_lstm/{fd}/seed_{s}/raw_predictions.json", "CNN_LSTM"),
    "gated":     ("results/gated_fusion/{fd}/seed_{s}/raw_predictions.json", "Gated_Fusion"),
}
ERR = {m: {} for m in SRC}
for m, (tp, key) in SRC.items():
    for fd in DATASETS:
        rows, y0 = [], None
        for s in SEEDS:
            d = json.load(open(tp.format(fd=fd, s=s)))[key]
            yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
            y0 = yt if y0 is None else y0
            assert np.array_equal(yt, y0)
            rows.append(np.abs(yp - yt))
        ERR[m][fd] = np.array(rows)
COMPARE = [("noncausal", "plain"), ("causal", "plain"), ("ma", "plain"), ("ema", "plain"),
           ("cnn_lstm", "plain"), ("gated", "plain"), ("ma", "causal")]

def boot(D, mode):
    S, E = D.shape
    out = np.empty(B)
    for b in range(B):
        si = rng.integers(0, S, S) if mode in ("both", "seeds") else np.arange(S)
        ei = rng.integers(0, E, E) if mode in ("both", "engines") else np.arange(E)
        out[b] = D[si].mean(0)[ei].mean()
    return out

def varcomp(D):
    """Mean, SE and Satterthwaite df of the grand mean for a crossed seeds x engines table."""
    S, E = D.shape; m = D.mean()
    r, c = D.mean(1), D.mean(0)
    ms_s = E * ((r - m) ** 2).sum() / (S - 1)
    ms_e = S * ((c - m) ** 2).sum() / (E - 1)
    ms_r = ((D - r[:, None] - c[None, :] + m) ** 2).sum() / ((S - 1) * (E - 1))
    # Var(mean) = sigma2_seed/S + MS_eng/(S*E); the seed component is floored at zero (an unbiased
    # estimate can go negative, which would understate uncertainty)
    s2_seed = max(0.0, (ms_s - ms_r) / E)
    if s2_seed > 0:
        a = np.array([ms_s, ms_e, -ms_r]) / (S * E)
        dfs = np.array([S - 1, E - 1, (S - 1) * (E - 1)])
        var = a.sum(); df = var ** 2 / ((a ** 2) / dfs).sum()
    else:
        var = ms_e / (S * E); df = E - 1
    var = max(var, 1e-12)
    share_seed = max(0.0, (ms_s - ms_r) / E) / S / var
    return m, np.sqrt(var), max(df, 1.0), share_seed

def tost_t(m, se, df, delta):
    return max(1 - stats.t.cdf((m + delta) / se, df), stats.t.cdf((m - delta) / se, df))

res = {}
print(f"Two-level uncertainty, {B} bootstrap replicates; equivalence margin {DELTA} cycles (|err|)\n")
for A, Bm in COMPARE:
    rows = []
    for fd in DATASETS:
        D = ERR[A][fd] - ERR[Bm][fd]
        m = D.mean()
        bb, bs, be = boot(D, "both"), boot(D, "seeds"), boot(D, "engines")
        q = lambda x, p: float(np.percentile(x, p))
        m_vc, se, df, share = varcomp(D)
        tc = stats.t.ppf(0.95, df); t975 = stats.t.ppf(0.975, df)
        ci_vc90 = (m_vc - tc * se, m_vc + tc * se)
        ci_b90 = (q(bb, 5), q(bb, 95))
        lo, hi = min(ci_vc90[0], ci_b90[0]), max(ci_vc90[1], ci_b90[1])   # conservative union
        p_vc_zero = 2 * (1 - stats.t.cdf(abs(m_vc) / se, df))
        pb_zero = min(1.0, 2 * (min((bb <= 0).mean(), (bb >= 0).mean()) * B + 1) / (B + 1))
        plo, phi = ((bb <= -DELTA).sum() + 1) / (B + 1), ((bb >= DELTA).sum() + 1) / (B + 1)
        rows.append(dict(fd=fd, mean_diff=float(m),
            sd_seeds_only=float(bs.std()), sd_engines_only=float(be.std()), sd_both=float(bb.std()),
            share_of_variance_from_seeds=float(share), df=float(df),
            ci90_bootstrap=ci_b90, ci90_varcomp=ci_vc90, ci90_conservative=(lo, hi),
            ci95_varcomp=(m_vc - t975 * se, m_vc + t975 * se),
            min_margin=float(max(abs(lo), abs(hi))),
            p_tost_bootstrap=float(max(plo, phi)), p_tost_varcomp=float(tost_t(m_vc, se, df, DELTA)),
            p_zero_bootstrap=float(pb_zero), p_zero_varcomp=float(p_vc_zero)))
    for key_b, key_v in (("p_tost_bootstrap", "p_tost_varcomp"),):
        for r, pa in zip(rows, multipletests([max(x["p_tost_bootstrap"], x["p_tost_varcomp"]) for x in rows], method="holm")[1]):
            r["p_tost_holm_conservative"] = float(pa)
    for r, pa in zip(rows, multipletests([max(x["p_zero_bootstrap"], x["p_zero_varcomp"]) for x in rows], method="holm")[1]):
        r["p_zero_holm_conservative"] = float(pa)
        r["verdict"] = ("equivalent, yet detectably different" if (r["p_tost_holm_conservative"] < ALPHA and pa < ALPHA)
                        else "equivalent" if r["p_tost_holm_conservative"] < ALPHA else "different" if pa < ALPHA else "inconclusive")
    res[f"{A}_vs_{Bm}"] = rows
    print(f"{A} vs {Bm}")
    print(f"  {'':6s}{'mean':>7s} {'sd:eng':>7s} {'sd:seed':>8s} {'sd:both':>8s} {'seed%':>6s} {'90% CI (conservative)':>24s} {'min margin':>11s} {'TOST p(H)':>10s} {'p!=0 (H)':>9s}  verdict")
    for r in rows:
        print(f"  {r['fd']:6s}{r['mean_diff']:+7.2f} {r['sd_engines_only']:7.2f} {r['sd_seeds_only']:8.2f} {r['sd_both']:8.2f} {100*r['share_of_variance_from_seeds']:5.0f}% "
              f"[{r['ci90_conservative'][0]:+6.2f},{r['ci90_conservative'][1]:+6.2f}] {r['min_margin']:11.2f} {r['p_tost_holm_conservative']:10.4f} {r['p_zero_holm_conservative']:9.4f}  {r['verdict']}")

# ---- seed-budget redo with the hierarchical test (k >= 2 seeds needed to estimate seed variance)
def hier_flags(errA, errB, subset):
    ps = []
    for fd in DATASETS:
        D = errA[fd][list(subset)] - errB[fd][list(subset)]
        m, se, df, _ = varcomp(D)
        ps.append(2 * (1 - stats.t.cdf(abs(m) / se, df)))
    raw = tuple(int(p < ALPHA) for p in ps)
    holm = tuple(int(x) for x in multipletests(ps, method="holm")[1] < ALPHA)
    return raw, holm

print("\nSeed budget with the hierarchical (seeds x engines) test: how often does a k-seed study flag a difference?")
sb = {}
for A, Bm in [("noncausal", "plain"), ("causal", "plain"), ("ma", "causal")]:
    full_raw, full_holm = hier_flags(ERR[A], ERR[Bm], range(6))
    print(f"\n{A} vs {Bm}: six-seed hierarchical: raw-sig {[d for d, f in zip(DATASETS, full_raw) if f]}, Holm-sig {[d for d, f in zip(DATASETS, full_holm) if f]}")
    print(f"  {'k':>2s} {'P(any raw sig)':>15s} {'P(any Holm sig)':>16s} {'P(Holm set = 6-seed)':>21s}")
    out = []
    for k in range(2, 7):
        F = [hier_flags(ERR[A], ERR[Bm], s) for s in itertools.combinations(range(6), k)]
        a, h, e = np.mean([any(r) for r, _ in F]), np.mean([any(x) for _, x in F]), np.mean([x == full_holm for _, x in F])
        out.append(dict(k=k, p_any_raw=float(a), p_any_holm=float(h), p_holm_set_equal=float(e)))
        print(f"  {k:>2d} {a:15.2f} {h:16.2f} {e:21.2f}")
    sb[f"{A}_vs_{Bm}"] = dict(full_raw=full_raw, full_holm=full_holm, by_k=out)

json.dump(dict(B=B, delta=DELTA, comparisons=res, seed_budget_hierarchical=sb), open("results/hier_bootstrap.json", "w"), indent=1, default=float)
print("\nsaved results/hier_bootstrap.json")
