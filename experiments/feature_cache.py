"""
Dataset-level cache for ARIMA trend features.

The trend features are a deterministic function of the regime-normalised sensor series, which
does not depend on the training seed (the K-means partition is seed-stable and per-regime
statistics are label-invariant). Recomputing them for every seed is therefore pure waste.
Reuse is guarded by a SHA-256 fingerprint of the normalised series: if a run's normalised data
ever differed, the cache is rejected and the features are recomputed.
"""
import hashlib, json, os
import numpy as np
import pandas as pd

CACHE_ROOT = "results/_feature_cache"


def fingerprint(train_raw, test_raw, feature_cols):
    h = hashlib.sha256()
    for df in (train_raw, test_raw):
        h.update(np.round(df[feature_cols].to_numpy(dtype=np.float64), 9).tobytes())
        h.update(df["unit_number"].to_numpy().tobytes())
        h.update(df["time_cycles"].to_numpy().tobytes())
    return h.hexdigest()


def load_or_build(variant, fd, train_raw, test_raw, feature_cols, build_fn, get_counters):
    """
    build_fn(df) -> df with trend columns added. get_counters() -> JSON-able dict of counters
    accumulated by build_fn (fits, non-converged fits, fallbacks, ...).
    Returns (train_arima, test_arima, counters, from_cache).
    """
    os.makedirs(CACHE_ROOT, exist_ok=True)
    stem = f"{CACHE_ROOT}/{variant}_{fd}"
    fp = fingerprint(train_raw, test_raw, feature_cols)
    if all(os.path.exists(f"{stem}{s}") for s in ("_train.parquet", "_test.parquet", "_meta.json")):
        meta = json.load(open(f"{stem}_meta.json"))
        if meta["fingerprint"] == fp:
            print(f"[feature_cache] {variant}/{fd}: cache hit, fingerprint verified")
            return (pd.read_parquet(f"{stem}_train.parquet"), pd.read_parquet(f"{stem}_test.parquet"),
                    meta["counters"], True)
        print(f"[feature_cache] {variant}/{fd}: FINGERPRINT MISMATCH, recomputing")
    train_arima = build_fn(train_raw)
    test_arima = build_fn(test_raw)
    counters = get_counters()
    tmp = f"{stem}.tmp{os.getpid()}"
    train_arima.to_parquet(f"{tmp}_train.parquet"); test_arima.to_parquet(f"{tmp}_test.parquet")
    json.dump({"fingerprint": fp, "counters": counters}, open(f"{tmp}_meta.json", "w"), indent=1)
    for s in ("_train.parquet", "_test.parquet", "_meta.json"):
        os.replace(f"{tmp}{s}", f"{stem}{s}")
    return train_arima, test_arima, counters, False
