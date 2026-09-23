"""
Unified pipeline for one C-MAPSS sub-dataset (FD001 / FD002 / FD003 / FD004).
Real experiment, real data, real results -- run end to end per dataset.

Usage: python3 run_dataset.py FD001 [seed]   (or FD002, FD003, FD004)
"""
import sys
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FD = sys.argv[1] if len(sys.argv) > 1 else "FD001"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}[FD]
RUL_CAP = 125
WINDOW = 30
SENSOR_VAR_THRESH = 1e-4

OUT_DIR = f"results/order_selected/{FD}/seed_{SEED}"
os.makedirs(OUT_DIR, exist_ok=True)

t_start = time.time()
print(f"===== Running {FD} seed={SEED} (N_REGIMES={N_REGIMES}) =====")

# ---------- 1. Load ----------
train_raw = pd.read_parquet(f"data/train_{FD}.parquet")
test_raw = pd.read_parquet(f"data/valid_{FD}.parquet")

for c in train_raw.columns:
    if c.startswith("s_"):
        train_raw[c] = train_raw[c].astype(float)
        test_raw[c] = test_raw[c].astype(float)

sensor_cols_all = [c for c in train_raw.columns if c.startswith("s_")]
variances = train_raw[sensor_cols_all].std()
sensor_cols = sorted(variances[variances > SENSOR_VAR_THRESH].index.tolist(), key=lambda x: int(x.split("_")[1]))
print(f"Selected {len(sensor_cols)} informative sensors: {sensor_cols}")
feature_cols = sensor_cols

# ---------- 2. RUL capping ----------
train_raw["RUL_capped"] = train_raw["RUL"].clip(upper=RUL_CAP)
test_raw["RUL_capped"] = test_raw["RUL"].clip(upper=RUL_CAP)

# ---------- 3. Operating-regime clustering + per-regime normalization ----------
# K-means on the 3 operational settings (Li, Ding & Sun 2018). For FD001 (single
# condition), N_REGIMES=1 reduces this to ordinary global normalization; the same
# code path is used for all three sub-datasets for consistency.
setting_cols = ["setting_1", "setting_2", "setting_3"]
kmeans = KMeans(n_clusters=N_REGIMES, random_state=SEED, n_init=10)
train_raw["regime"] = kmeans.fit_predict(train_raw[setting_cols])
test_raw["regime"] = kmeans.predict(test_raw[setting_cols])
print("Regime sizes (train):", train_raw["regime"].value_counts().to_dict())

regime_stats = {}
for r in range(N_REGIMES):
    mask = train_raw["regime"] == r
    mu = train_raw.loc[mask, feature_cols].mean()
    sigma = train_raw.loc[mask, feature_cols].std().replace(0, 1.0)
    regime_stats[r] = (mu, sigma)

def regime_normalize(df):
    out = df.copy()
    for r in range(N_REGIMES):
        mask = out["regime"] == r
        mu, sigma = regime_stats[r]
        out.loc[mask, feature_cols] = (out.loc[mask, feature_cols] - mu) / sigma
    return out

train_raw = regime_normalize(train_raw)
test_raw = regime_normalize(test_raw)
# clip to control outliers from any regime with few samples
train_raw[feature_cols] = train_raw[feature_cols].clip(-6, 6)
test_raw[feature_cols] = test_raw[feature_cols].clip(-6, 6)

# ---------- 4. ARIMA trend-feature extraction per unit per sensor, WITH per-channel order selection ----------
CANDIDATE_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]
order_choice_counts = {}

def arima_trend_features_order_selected(df, feature_cols, candidate_orders=CANDIDATE_ORDERS):
    out = df.copy()
    for col in feature_cols:
        out[f"{col}_trend"] = np.nan
    t0 = time.time()
    units = df["unit_number"].unique()
    n_units = len(units)
    for i, unit in enumerate(units):
        idx = df.index[df["unit_number"] == unit]
        g = df.loc[idx]
        for col in feature_cols:
            series = g[col].values
            best_aic = np.inf
            best_fitted = None
            best_order = None
            for order in candidate_orders:
                try:
                    fit = ARIMA(series, order=order).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fitted = fit.predict(start=0, end=len(series) - 1)
                        best_order = order
                except Exception:
                    continue
            if best_fitted is None:
                best_fitted = pd.Series(series).rolling(5, min_periods=1).mean().values
                best_order = "fallback"
            out.loc[idx, f"{col}_trend"] = best_fitted
            order_choice_counts[best_order] = order_choice_counts.get(best_order, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  ARIMA order-selected trend extraction: {i+1}/{n_units} units done ({time.time()-t0:.1f}s)")
    return out

print("Fitting per-unit-per-sensor ARIMA trend models with order selection (train)...")
train_arima = arima_trend_features_order_selected(train_raw, feature_cols)
print("Fitting per-unit-per-sensor ARIMA trend models with order selection (test)...")
test_arima = arima_trend_features_order_selected(test_raw, feature_cols)
print("Order selection counts (train+test combined):", order_choice_counts)

trend_cols = [f"{c}_trend" for c in feature_cols]
train_arima[trend_cols] = train_arima[trend_cols].bfill().ffill()
test_arima[trend_cols] = test_arima[trend_cols].bfill().ffill()
print(f"ARIMA stage done at {time.time()-t_start:.1f}s")

train_arima.to_parquet(f"{OUT_DIR}/_arima_cache_train.parquet")
test_arima.to_parquet(f"{OUT_DIR}/_arima_cache_test.parquet")

# ---------- 5. Windowing ----------
hybrid_feats = feature_cols + trend_cols

all_units = sorted(train_arima["unit_number"].unique())
rng = np.random.RandomState(SEED)
n_val = max(10, int(0.15 * len(all_units)))
val_units = set(rng.choice(all_units, size=n_val, replace=False))
tr_units = [u for u in all_units if u not in val_units]
test_units = sorted(test_arima["unit_number"].unique())

def make_windows(df, feats, units, window=WINDOW, stride=1, label_col="RUL_capped"):
    X, y = [], []
    for u in units:
        g = df[df["unit_number"] == u].sort_values("time_cycles")
        arr = g[feats].values
        labels = g[label_col].values
        L = len(arr)
        if L < window:
            pad = np.repeat(arr[0:1], window - L, axis=0)
            arr = np.vstack([pad, arr])
            labels = np.concatenate([np.repeat(labels[0], window - L), labels])
            L = window
        for end in range(window, L + 1, stride):
            X.append(arr[end - window:end])
            y.append(labels[end - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_last_window(df, feats, units, window=WINDOW, label_col="RUL_capped"):
    X, y = [], []
    for u in units:
        g = df[df["unit_number"] == u].sort_values("time_cycles")
        arr = g[feats].values
        L = len(arr)
        if L < window:
            pad = np.repeat(arr[0:1], window - L, axis=0)
            arr = np.vstack([pad, arr])
        X.append(arr[-window:])
        y.append(g[label_col].values[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def per_cycle_table(df, feats, units, label_col="RUL_capped"):
    sub = df[df["unit_number"].isin(units)]
    return sub[feats + [label_col]].reset_index(drop=True)

def last_cycle_table(df, feats, units, label_col="RUL_capped"):
    rows = []
    for u in units:
        g = df[df["unit_number"] == u].sort_values("time_cycles")
        rows.append(list(g[feats].iloc[-1].values) + [g[label_col].iloc[-1]])
    return pd.DataFrame(rows, columns=feats + [label_col])

results = {}

# ---------- 6. Random Forest baseline ----------
rf_train_tbl = per_cycle_table(train_arima, feature_cols, tr_units)
rf_test_tbl = last_cycle_table(test_arima, feature_cols, test_units)
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1)
rf.fit(rf_train_tbl[feature_cols], rf_train_tbl["RUL_capped"])
rf_pred = rf.predict(rf_test_tbl[feature_cols])
results["RandomForest_baseline"] = {"y_true": rf_test_tbl["RUL_capped"].tolist(), "y_pred": rf_pred.tolist()}
print(f"RF baseline trained on {len(rf_train_tbl)} rows.")

# ---------- 7. LSTM models ----------
class RULLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def train_lstm(X_tr, y_tr, X_val, y_val, n_features, epochs=60, batch_size=128, lr=1e-3, patience=8, tag=""):
    model = RULLSTM(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    X_val_t, y_val_t = torch.tensor(X_val).to(device), torch.tensor(y_val).to(device)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = lossfn(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{FD}:{tag}] epoch {epoch+1}/{epochs} train_mse={train_loss:.2f} val_rmse={val_loss**0.5:.2f}")
        if bad_epochs >= patience:
            print(f"[{FD}:{tag}] early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model

Xtr_raw, ytr_raw = make_windows(train_arima, feature_cols, tr_units)
Xval_raw, yval_raw = make_windows(train_arima, feature_cols, list(val_units))
Xtest_raw, ytest_raw = make_last_window(test_arima, feature_cols, test_units)
print(f"plain-LSTM windows: train={Xtr_raw.shape} val={Xval_raw.shape} test={Xtest_raw.shape}")

model_plain = train_lstm(Xtr_raw, ytr_raw, Xval_raw, yval_raw, len(feature_cols), tag="plain")
with torch.no_grad():
    pred_plain = model_plain(torch.tensor(Xtest_raw)).numpy()
results["Plain_LSTM"] = {"y_true": ytest_raw.tolist(), "y_pred": pred_plain.tolist()}

Xtr_hyb, ytr_hyb = make_windows(train_arima, hybrid_feats, tr_units)
Xval_hyb, yval_hyb = make_windows(train_arima, hybrid_feats, list(val_units))
Xtest_hyb, ytest_hyb = make_last_window(test_arima, hybrid_feats, test_units)

model_hybrid = train_lstm(Xtr_hyb, ytr_hyb, Xval_hyb, yval_hyb, len(hybrid_feats), tag="hybrid")
with torch.no_grad():
    pred_hybrid = model_hybrid(torch.tensor(Xtest_hyb)).numpy()
results["Hybrid_ARIMA_LSTM"] = {"y_true": ytest_hyb.tolist(), "y_pred": pred_hybrid.tolist()}

# ---------- 8. Evaluation ----------
def nasa_phm_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))

summary = {}
for name, r in results.items():
    yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    score = nasa_phm_score(yt, yp)
    summary[name] = {"RMSE": rmse, "MAE": mae, "PHM_score": score, "n_test": len(yt)}
    print(f"{FD} {name}: RMSE={rmse:.3f} MAE={mae:.3f} PHM_score={score:.1f} n={len(yt)}")

summary["_meta"] = {
    "dataset": FD,
    "seed": SEED,
    "n_regimes": N_REGIMES,
    "n_train_units": len(tr_units),
    "n_val_units": len(val_units),
    "n_test_units": len(test_units),
    "n_train_windows": int(Xtr_raw.shape[0]),
    "runtime_sec": time.time() - t_start,
    "order_choice_counts": {str(k): v for k, v in order_choice_counts.items()},
}

with open(f"{OUT_DIR}/summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{OUT_DIR}/raw_predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"===== {FD} done in {time.time()-t_start:.1f}s =====")
