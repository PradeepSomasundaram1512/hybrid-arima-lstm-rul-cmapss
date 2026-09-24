#!/bin/bash
# Reproduce every experiment. Run from the repository root with the versions in requirements.txt.
#   pip install -r requirements.txt
# Stage 1 (about 1 hour): build and verify the ARIMA feature caches (seed 42 through every variant).
# Stage 2 (hours): all seeds for every variant, resumable:  python3 code/run_extra_seeds.py --workers 4
# Stage 3 (minutes): statistics:  python3 code/analysis_extended.py
# The original six seeds (42, 123, 2024, 7, 99, 314) can also be run one at a time, e.g.
#   python3 code/run_dataset.py FD001 42            # plain LSTM, RF, fixed-order hybrid
#   python3 code/run_dataset_order_selected.py FD001 42
#   python3 code/run_dataset_causal.py FD001 42
#   python3 code/run_trend_controls.py FD001 42
#   python3 code/run_capacity_controls.py FD001 42
set -e
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for FD in FD001 FD002 FD003 FD004; do
  python3 code/run_dataset.py $FD 42
  python3 code/run_dataset_order_selected.py $FD 42
  python3 code/run_dataset_causal.py $FD 42
done
python3 code/run_extra_seeds.py --workers 4
python3 code/analysis_extended.py
