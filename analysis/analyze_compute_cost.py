import json
import numpy as np

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 123, 2024, 7, 99, 314]

print(f"{'Dataset':8s} {'FixedOrder(s)':>15s} {'SelectedOrder(s)':>18s} {'Overhead':>10s}")
for fd in DATASETS:
    fixed_times, sel_times = [], []
    for seed in SEEDS:
        d_fix = json.load(open(f"results/{fd}/seed_{seed}/summary_metrics.json"))
        d_sel = json.load(open(f"results/order_selected/{fd}/seed_{seed}/summary_metrics.json"))
        fixed_times.append(d_fix["_meta"]["runtime_sec"])
        sel_times.append(d_sel["_meta"]["runtime_sec"])
    fm, sm = np.mean(fixed_times), np.mean(sel_times)
    print(f"{fd:8s} {fm:15.1f} {sm:18.1f} {sm/fm:9.2f}x")

# Rough per-model parameter count and inference cost note (architecture is identical
# except input width, which only affects the first LSTM layer's input weight matrix).
plain_features = {"FD001": 15, "FD002": 21, "FD003": 15, "FD004": 21}
hidden = 64
for fd in DATASETS:
    n_in_plain = plain_features[fd]
    n_in_hybrid = 2 * plain_features[fd]
    # LSTM param count: 4 * (hidden*(input+hidden) + hidden) per layer, 2 layers (2nd layer input=hidden)
    def lstm_params(n_in, hidden, layers=2):
        p = 4 * (hidden * (n_in + hidden) + hidden)  # layer 1
        for _ in range(layers - 1):
            p += 4 * (hidden * (hidden + hidden) + hidden)  # subsequent layers
        return p
    p_plain = lstm_params(n_in_plain, hidden)
    p_hybrid = lstm_params(n_in_hybrid, hidden)
    print(f"{fd}: plain LSTM params (approx, LSTM layers only) = {p_plain:,}  "
          f"hybrid = {p_hybrid:,}  (+{100*(p_hybrid-p_plain)/p_plain:.1f}%)")
