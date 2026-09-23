"""
Strong modern baseline: a 1D-CNN feature extractor followed by the same
2-layer LSTM used throughout this paper, representative of the CNN-LSTM
hybrid family common in recent C-MAPSS RUL literature (e.g., the
Muthukumar & Philip 2024 architecture we cite). Uses only raw sensor
features (no ARIMA trend), trained under the identical protocol (seed,
window, epochs, patience, optimizer) as the plain and hybrid LSTM models,
so it is directly comparable to Table III.

Usage: python3 run_cnn_lstm_baseline.py FD001 [seed]
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

OUT_DIR = f"results/cnn_lstm/{FD}/seed_{SEED}"
os.makedirs(OUT_DIR, exist_ok=True)

t_start = time.time()
print(f"===== CNN-LSTM baseline: {FD} seed={SEED} =====")

# ---------- Load and preprocess (identical to run_dataset.py, minus ARIMA) ----------
train_raw = pd.read_parquet(f"data/train_{FD}.parquet")
test_raw = pd.read_parquet(f"data/valid_{FD}.parquet")
for c in train_raw.columns:
    if c.startswith("s_"):
        train_raw[c] = train_raw[c].astype(float)
        test_raw[c] = test_raw[c].astype(float)

sensor_cols_all = [c for c in train_raw.columns if c.startswith("s_")]
variances = train_raw[sensor_cols_all].std()
feature_cols = sorted(variances[variances > SENSOR_VAR_THRESH].index.tolist(), key=lambda x: int(x.split("_")[1]))

train_raw["RUL_capped"] = train_raw["RUL"].clip(upper=RUL_CAP)
test_raw["RUL_capped"] = test_raw["RUL"].clip(upper=RUL_CAP)

setting_cols = ["setting_1", "setting_2", "setting_3"]
kmeans = KMeans(n_clusters=N_REGIMES, random_state=SEED, n_init=10)
train_raw["regime"] = kmeans.fit_predict(train_raw[setting_cols])
test_raw["regime"] = kmeans.predict(test_raw[setting_cols])

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
train_raw[feature_cols] = train_raw[feature_cols].clip(-6, 6)
test_raw[feature_cols] = test_raw[feature_cols].clip(-6, 6)

all_units = sorted(train_raw["unit_number"].unique())
rng = np.random.RandomState(SEED)
n_val = max(10, int(0.15 * len(all_units)))
val_units = set(rng.choice(all_units, size=n_val, replace=False))
tr_units = [u for u in all_units if u not in val_units]
test_units = sorted(test_raw["unit_number"].unique())

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

Xtr, ytr = make_windows(train_raw, feature_cols, tr_units)
Xval, yval = make_windows(train_raw, feature_cols, list(val_units))
Xtest, ytest = make_last_window(test_raw, feature_cols, test_units)
print(f"windows: train={Xtr.shape} val={Xval.shape} test={Xtest.shape}")

# ---------- CNN-LSTM model ----------
class CNNLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, cnn_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels * 2, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: (batch, time, features) -> conv over time axis
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, time, cnn_channels*2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def train_model(X_tr, y_tr, X_val, y_val, n_features, epochs=60, batch_size=128, lr=1e-3, patience=8):
    model = CNNLSTM(n_features).to(device)
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
        if (epoch + 1) % 10 == 0:
            print(f"[{FD}:cnn_lstm] epoch {epoch+1}/{epochs} val_rmse={val_loss**0.5:.2f}")
        if bad_epochs >= patience:
            print(f"[{FD}:cnn_lstm] early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model

model = train_model(Xtr, ytr, Xval, yval, len(feature_cols))
with torch.no_grad():
    pred = model(torch.tensor(Xtest)).numpy()

def nasa_phm_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))

rmse = float(np.sqrt(mean_squared_error(ytest, pred)))
mae = float(mean_absolute_error(ytest, pred))
score = nasa_phm_score(ytest, pred)
print(f"{FD} CNN_LSTM: RMSE={rmse:.3f} MAE={mae:.3f} PHM_score={score:.1f} n={len(ytest)}")

summary = {
    "CNN_LSTM": {"RMSE": rmse, "MAE": mae, "PHM_score": score, "n_test": len(ytest)},
    "_meta": {"dataset": FD, "seed": SEED, "runtime_sec": time.time() - t_start,
              "n_features": len(feature_cols), "n_train_windows": int(Xtr.shape[0])},
}
with open(f"{OUT_DIR}/summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{OUT_DIR}/raw_predictions.json", "w") as f:
    json.dump({"CNN_LSTM": {"y_true": ytest.tolist(), "y_pred": pred.tolist()}}, f, indent=2)

print(f"===== {FD} CNN-LSTM done in {time.time()-t_start:.1f}s =====")
