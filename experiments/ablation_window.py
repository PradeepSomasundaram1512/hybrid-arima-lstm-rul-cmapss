"""
Window-length ablation on FD001: does the hybrid model's advantage over the
plain LSTM change as the input window shrinks? Real experiment, reuses the
same ARIMA-augmented FD001 data already computed by run_dataset.py FD001
(cached to results/FD001/_arima_cache.parquet by this script on first run).
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

RUL_CAP = 125
SENSOR_VAR_THRESH = 1e-4
WINDOWS = [10, 20, 30, 50]

OUT_DIR = f"results/ablation_window/seed_{SEED}"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---- reuse FD001 ARIMA-augmented data if already cached by run_dataset.py ----
cache_path = f"results/FD001/seed_{SEED}/_arima_cache_train.parquet"
cache_path_test = f"results/FD001/seed_{SEED}/_arima_cache_test.parquet"

if os.path.exists(cache_path) and os.path.exists(cache_path_test):
    print("Loading cached ARIMA-augmented FD001 data...")
    train_arima = pd.read_parquet(cache_path)
    test_arima = pd.read_parquet(cache_path_test)
    all_sensor_cols = [c for c in train_arima.columns if c.startswith("s_") and "_trend" not in c]
    trend_cols_present = set(c for c in train_arima.columns if c.endswith("_trend"))
    feature_cols = [c for c in all_sensor_cols if f"{c}_trend" in trend_cols_present]
    trend_cols = [f"{c}_trend" for c in feature_cols]
else:
    print("No cache found -- recomputing ARIMA-augmented FD001 data...")
    train_raw = pd.read_parquet("data/train_FD001.parquet")
    test_raw = pd.read_parquet("data/valid_FD001.parquet")
    for c in train_raw.columns:
        if c.startswith("s_"):
            train_raw[c] = train_raw[c].astype(float)
            test_raw[c] = test_raw[c].astype(float)
    sensor_cols_all = [c for c in train_raw.columns if c.startswith("s_")]
    variances = train_raw[sensor_cols_all].std()
    feature_cols = sorted(variances[variances > SENSOR_VAR_THRESH].index.tolist(), key=lambda x: int(x.split("_")[1]))

    train_raw["RUL_capped"] = train_raw["RUL"].clip(upper=RUL_CAP)
    test_raw["RUL_capped"] = test_raw["RUL"].clip(upper=RUL_CAP)

    kmeans = KMeans(n_clusters=1, random_state=SEED, n_init=10)
    train_raw["regime"] = kmeans.fit_predict(train_raw[["setting_1", "setting_2", "setting_3"]])
    test_raw["regime"] = kmeans.predict(test_raw[["setting_1", "setting_2", "setting_3"]])
    mu = train_raw[feature_cols].mean()
    sigma = train_raw[feature_cols].std().replace(0, 1.0)
    train_raw[feature_cols] = ((train_raw[feature_cols] - mu) / sigma).clip(-6, 6)
    test_raw[feature_cols] = ((test_raw[feature_cols] - mu) / sigma).clip(-6, 6)

    def arima_trend_features(df, feature_cols, order=(1, 1, 0)):
        out = df.copy()
        for col in feature_cols:
            out[f"{col}_trend"] = np.nan
        for unit in df["unit_number"].unique():
            idx = df.index[df["unit_number"] == unit]
            g = df.loc[idx]
            for col in feature_cols:
                series = g[col].values
                try:
                    fit = ARIMA(series, order=order).fit()
                    fitted = fit.predict(start=0, end=len(series) - 1)
                except Exception:
                    fitted = pd.Series(series).rolling(5, min_periods=1).mean().values
                out.loc[idx, f"{col}_trend"] = fitted
        return out

    train_arima = arima_trend_features(train_raw, feature_cols)
    test_arima = arima_trend_features(test_raw, feature_cols)
    trend_cols = [f"{c}_trend" for c in feature_cols]
    train_arima[trend_cols] = train_arima[trend_cols].bfill().ffill()
    test_arima[trend_cols] = test_arima[trend_cols].bfill().ffill()
    train_arima.to_parquet(cache_path)
    test_arima.to_parquet(cache_path_test)

hybrid_feats = feature_cols + trend_cols
all_units = sorted(train_arima["unit_number"].unique())
rng = np.random.RandomState(SEED)
val_units = set(rng.choice(all_units, size=15, replace=False))
tr_units = [u for u in all_units if u not in val_units]
test_units = sorted(test_arima["unit_number"].unique())


def make_windows(df, feats, units, window, stride=1, label_col="RUL_capped"):
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


def make_last_window(df, feats, units, window, label_col="RUL_capped"):
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


class RULLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_lstm(X_tr, y_tr, X_val, y_val, n_features, epochs=50, batch_size=128, lr=1e-3, patience=6, tag=""):
    model = RULLSTM(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    X_val_t, y_val_t = torch.tensor(X_val).to(device), torch.tensor(y_val).to(device)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = lossfn(model(X_val_t), y_val_t).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model


def nasa_phm_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


ablation_results = {}
for W in WINDOWS:
    t0 = time.time()
    print(f"--- window={W} ---")
    Xtr_p, ytr_p = make_windows(train_arima, feature_cols, tr_units, W)
    Xval_p, yval_p = make_windows(train_arima, feature_cols, list(val_units), W)
    Xtest_p, ytest_p = make_last_window(test_arima, feature_cols, test_units, W)
    m_plain = train_lstm(Xtr_p, ytr_p, Xval_p, yval_p, len(feature_cols), tag=f"plain-W{W}")
    with torch.no_grad():
        pred_plain = m_plain(torch.tensor(Xtest_p)).numpy()

    Xtr_h, ytr_h = make_windows(train_arima, hybrid_feats, tr_units, W)
    Xval_h, yval_h = make_windows(train_arima, hybrid_feats, list(val_units), W)
    Xtest_h, ytest_h = make_last_window(test_arima, hybrid_feats, test_units, W)
    m_hybrid = train_lstm(Xtr_h, ytr_h, Xval_h, yval_h, len(hybrid_feats), tag=f"hybrid-W{W}")
    with torch.no_grad():
        pred_hybrid = m_hybrid(torch.tensor(Xtest_h)).numpy()

    rmse_plain = float(np.sqrt(mean_squared_error(ytest_p, pred_plain)))
    mae_plain = float(mean_absolute_error(ytest_p, pred_plain))
    score_plain = nasa_phm_score(ytest_p, pred_plain)

    rmse_hybrid = float(np.sqrt(mean_squared_error(ytest_h, pred_hybrid)))
    mae_hybrid = float(mean_absolute_error(ytest_h, pred_hybrid))
    score_hybrid = nasa_phm_score(ytest_h, pred_hybrid)

    ablation_results[W] = {
        "plain": {"RMSE": rmse_plain, "MAE": mae_plain, "PHM_score": score_plain},
        "hybrid": {"RMSE": rmse_hybrid, "MAE": mae_hybrid, "PHM_score": score_hybrid},
    }
    print(f"W={W}: plain RMSE={rmse_plain:.2f} | hybrid RMSE={rmse_hybrid:.2f} "
          f"| delta={(rmse_plain-rmse_hybrid):.2f} ({time.time()-t0:.1f}s)")

with open(f"{OUT_DIR}/ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print("Saved", f"{OUT_DIR}/ablation_results.json")
