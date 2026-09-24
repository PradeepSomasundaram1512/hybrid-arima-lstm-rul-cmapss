"""
Tier 2 mechanistic analysis: does FD002 have more "linear" (ARIMA-explainable)
sensor trends than FD001/FD003/FD004, which would give a mechanistic reason
why a well-fit ARIMA trend feature helps specifically there?

For each dataset, fit ARIMA(0,1,1) (the AIC-dominant order across all four
datasets) per (unit, sensor) on a sample of training units, and compute the
fraction of the first-differenced signal's variance explained by the fitted
trend: R^2 = 1 - var(residual) / var(diff(raw signal)). Average across
sensors and sampled units per dataset. Higher = more of the signal is
explainable by a simple linear/MA trend model.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SENSOR_VAR_THRESH = 1e-4
N_SAMPLE_UNITS = 25
SEED = 42

results = {}
for fd in DATASETS:
    train_raw = pd.read_parquet(f"data/train_{fd}.parquet")
    for c in train_raw.columns:
        if c.startswith("s_"):
            train_raw[c] = train_raw[c].astype(float)
    sensor_cols_all = [c for c in train_raw.columns if c.startswith("s_")]
    variances = train_raw[sensor_cols_all].std()
    feature_cols = sorted(variances[variances > SENSOR_VAR_THRESH].index.tolist(), key=lambda x: int(x.split("_")[1]))

    kmeans = KMeans(n_clusters=N_REGIMES[fd], random_state=SEED, n_init=10)
    train_raw["regime"] = kmeans.fit_predict(train_raw[["setting_1", "setting_2", "setting_3"]])
    for r in range(N_REGIMES[fd]):
        mask = train_raw["regime"] == r
        mu = train_raw.loc[mask, feature_cols].mean()
        sigma = train_raw.loc[mask, feature_cols].std().replace(0, 1.0)
        train_raw.loc[mask, feature_cols] = ((train_raw.loc[mask, feature_cols] - mu) / sigma).clip(-6, 6)

    units = sorted(train_raw["unit_number"].unique())
    rng = np.random.RandomState(SEED)
    sample_units = rng.choice(units, size=min(N_SAMPLE_UNITS, len(units)), replace=False)

    r2_values = []
    r2_units = []
    for unit in sample_units:
        g = train_raw[train_raw["unit_number"] == unit].sort_values("time_cycles")
        for col in feature_cols:
            series = g[col].values
            if len(series) < 10:
                continue
            diffs = np.diff(series)
            var_diff = np.var(diffs)
            if var_diff < 1e-8:
                continue
            try:
                fit = ARIMA(series, order=(0, 1, 1)).fit()
                resid = fit.resid[1:]  # drop first differencing NaN/transient
                var_resid = np.var(resid)
                r2 = 1 - (var_resid / var_diff)
                r2_values.append(r2)
                r2_units.append(int(unit))
            except Exception:
                continue

    r2_arr = np.array(r2_values)
    unit_arr = np.array(r2_units)
    unit_means = np.array([r2_arr[unit_arr == u].mean() for u in np.unique(unit_arr)])
    results[fd] = {"mean_r2": float(np.mean(r2_arr)), "median_r2": float(np.median(r2_arr)),
                    "std_r2": float(np.std(r2_arr)), "n": len(r2_arr), "r2_values": r2_arr.tolist(),
                    "units": unit_arr.tolist(), "n_units": len(np.unique(unit_arr)),
                    "unit_means": unit_means.tolist()}
    print(f"{fd}: mean R^2={results[fd]['mean_r2']:.3f}  median={results[fd]['median_r2']:.3f}  "
          f"std={results[fd]['std_r2']:.3f}  (n={results[fd]['n']} unit-sensor fits, {len(sample_units)} units x {len(feature_cols)} sensors, "
          f"unit-level mean R^2={unit_means.mean():.3f}+/-{unit_means.std(ddof=1):.3f})")

import json
with open("results/trend_explainability.json", "w") as f:
    json.dump(results, f, indent=2)
