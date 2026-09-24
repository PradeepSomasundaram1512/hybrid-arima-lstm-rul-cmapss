"""
Gated-fusion hybrid: instead of naively concatenating the raw sensor window
with the ARIMA trend window (the fusion mechanism used throughout the rest of
this paper), pass both through a small learned gate that decides, per
timestep and per channel, how much of the trend feature to admit. This
directly targets the failure mode the paper's Discussion section hypothesizes:
a poorly-fit trend feature is not neutral, it actively misleads a naive
concatenation, and a gate should let the network learn to suppress it when it
is not informative rather than being stuck with a fixed contribution.

gate_t = sigmoid(W_g [raw_t, trend_t] + b_g)      (elementwise, per trend channel)
input_t = [raw_t, gate_t * trend_t]
followed by the same 2-layer LSTM used everywhere else in this paper.

Reuses the AIC-selected ARIMA trend features already cached by
run_dataset_order_selected.py (results/order_selected/{FD}/seed_{SEED}/_arima_cache_*.parquet)
since order selection, not a fixed order, is what makes the trend feature
informative (Section V-C) -- testing a new fusion mechanism on top of a
known-poor fixed-order trend feature would confound the two questions.

Usage: python3 run_gated_fusion.py FD001 [seed]
"""
import sys
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FD = sys.argv[1] if len(sys.argv) > 1 else "FD001"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

RUL_CAP = 125
WINDOW = 30

CACHE_DIR = f"results/order_selected/{FD}/seed_{SEED}"
OUT_DIR = f"results/gated_fusion/{FD}/seed_{SEED}"
os.makedirs(OUT_DIR, exist_ok=True)

t_start = time.time()
print(f"===== Gated-fusion hybrid: {FD} seed={SEED} =====")

train_arima = pd.read_parquet(f"{CACHE_DIR}/_arima_cache_train.parquet")
test_arima = pd.read_parquet(f"{CACHE_DIR}/_arima_cache_test.parquet")

all_sensor_cols = [c for c in train_arima.columns if c.startswith("s_") and not c.endswith("_trend")]
trend_cols_present = set(c[:-6] for c in train_arima.columns if c.endswith("_trend"))
feature_cols = sorted([c for c in all_sensor_cols if c in trend_cols_present], key=lambda x: int(x.split("_")[1]))
trend_cols = [f"{c}_trend" for c in feature_cols]
print(f"{len(feature_cols)} informative sensors with trend features: {feature_cols}")

all_units = sorted(train_arima["unit_number"].unique())
rng = np.random.RandomState(SEED)
n_val = max(10, int(0.15 * len(all_units)))
val_units = set(rng.choice(all_units, size=n_val, replace=False))
tr_units = [u for u in all_units if u not in val_units]
test_units = sorted(test_arima["unit_number"].unique())


def make_windows(df, raw_feats, trend_feats, units, window=WINDOW, stride=1, label_col="RUL_capped"):
    Xr, Xt, y = [], [], []
    for u in units:
        g = df[df["unit_number"] == u].sort_values("time_cycles")
        arr_r = g[raw_feats].values
        arr_t = g[trend_feats].values
        labels = g[label_col].values
        L = len(arr_r)
        if L < window:
            pad_r = np.repeat(arr_r[0:1], window - L, axis=0)
            pad_t = np.repeat(arr_t[0:1], window - L, axis=0)
            arr_r = np.vstack([pad_r, arr_r])
            arr_t = np.vstack([pad_t, arr_t])
            labels = np.concatenate([np.repeat(labels[0], window - L), labels])
            L = window
        for end in range(window, L + 1, stride):
            Xr.append(arr_r[end - window:end])
            Xt.append(arr_t[end - window:end])
            y.append(labels[end - 1])
    return np.array(Xr, dtype=np.float32), np.array(Xt, dtype=np.float32), np.array(y, dtype=np.float32)


def make_last_window(df, raw_feats, trend_feats, units, window=WINDOW, label_col="RUL_capped"):
    Xr, Xt, y = [], [], []
    for u in units:
        g = df[df["unit_number"] == u].sort_values("time_cycles")
        arr_r = g[raw_feats].values
        arr_t = g[trend_feats].values
        L = len(arr_r)
        if L < window:
            pad_r = np.repeat(arr_r[0:1], window - L, axis=0)
            pad_t = np.repeat(arr_t[0:1], window - L, axis=0)
            arr_r = np.vstack([pad_r, arr_r])
            arr_t = np.vstack([pad_t, arr_t])
        Xr.append(arr_r[-window:])
        Xt.append(arr_t[-window:])
        y.append(g[label_col].values[-1])
    return np.array(Xr, dtype=np.float32), np.array(Xt, dtype=np.float32), np.array(y, dtype=np.float32)


class GatedHybridLSTM(nn.Module):
    def __init__(self, n_raw, n_trend, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gate = nn.Linear(n_raw + n_trend, n_trend)
        self.lstm = nn.LSTM(n_raw + n_trend, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_raw, x_trend):
        gate_in = torch.cat([x_raw, x_trend], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))
        gated_trend = g * x_trend
        fused = torch.cat([x_raw, gated_trend], dim=-1)
        out, _ = self.lstm(fused)
        return self.head(out[:, -1, :]).squeeze(-1), g


def train_model(Xr_tr, Xt_tr, y_tr, Xr_val, Xt_val, y_val, n_raw, n_trend,
                 epochs=60, batch_size=128, lr=1e-3, patience=8):
    model = GatedHybridLSTM(n_raw, n_trend).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    tr_loader = DataLoader(
        TensorDataset(torch.tensor(Xr_tr), torch.tensor(Xt_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xr_val_t, Xt_val_t, y_val_t = torch.tensor(Xr_val).to(device), torch.tensor(Xt_val).to(device), torch.tensor(y_val).to(device)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xr, xt, yb in tr_loader:
            xr, xt, yb = xr.to(device), xt.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xr, xt)
            loss = lossfn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(Xr_val_t, Xt_val_t)
            val_loss = lossfn(val_pred, y_val_t).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if (epoch + 1) % 10 == 0:
            print(f"[{FD}:gated] epoch {epoch+1}/{epochs} val_rmse={val_loss**0.5:.2f}")
        if bad_epochs >= patience:
            print(f"[{FD}:gated] early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model


Xr_tr, Xt_tr, y_tr = make_windows(train_arima, feature_cols, trend_cols, tr_units)
Xr_val, Xt_val, y_val = make_windows(train_arima, feature_cols, trend_cols, list(val_units))
Xr_test, Xt_test, y_test = make_last_window(test_arima, feature_cols, trend_cols, test_units)
print(f"gated-fusion windows: train={Xr_tr.shape} val={Xr_val.shape} test={Xr_test.shape}")

model = train_model(Xr_tr, Xt_tr, y_tr, Xr_val, Xt_val, y_val, len(feature_cols), len(trend_cols))
with torch.no_grad():
    pred, gate_vals = model(torch.tensor(Xr_test), torch.tensor(Xt_test))
    pred = pred.numpy()
    mean_gate = float(gate_vals.mean().item())


def nasa_phm_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))
score = nasa_phm_score(y_test, pred)
print(f"{FD} Gated_Fusion: RMSE={rmse:.3f} MAE={mae:.3f} PHM_score={score:.1f} n={len(y_test)} mean_gate={mean_gate:.3f}")

summary = {
    "Gated_Fusion": {"RMSE": rmse, "MAE": mae, "PHM_score": score, "n_test": len(y_test)},
    "_meta": {
        "dataset": FD, "seed": SEED, "runtime_sec": time.time() - t_start,
        "n_raw_features": len(feature_cols), "n_trend_features": len(trend_cols),
        "n_train_windows": int(Xr_tr.shape[0]), "mean_test_gate_value": mean_gate,
    },
}
with open(f"{OUT_DIR}/summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{OUT_DIR}/raw_predictions.json", "w") as f:
    json.dump({"Gated_Fusion": {"y_true": y_test.tolist(), "y_pred": pred.tolist()}}, f, indent=2)

print(f"===== {FD} gated-fusion done in {time.time()-t_start:.1f}s =====")
