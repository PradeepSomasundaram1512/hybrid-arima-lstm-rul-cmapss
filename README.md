# A Hybrid ARIMA-LSTM Framework for Remaining Useful Life Prediction

Code, data, and results for the paper *"A Hybrid ARIMA-LSTM Framework for Remaining Useful Life Prediction in Industrial IoT Sensor Data: A Cross-Regime Study on NASA C-MAPSS"*, submitted to the 11th IEEE Special Session on Machine Learning on Big Data (MLBD 2026), IEEE BigData 2026.

## What this is

We test whether augmenting an LSTM with a per-sensor ARIMA trend feature improves remaining useful life (RUL) prediction on NASA's C-MAPSS turbofan degradation benchmark. Averaged over six random seeds across all four C-MAPSS sub-datasets, a fixed-order trend feature does not help, and we document how the specific significant result changed as we moved from three seeds to six. Under a per-channel, AIC-selected ARIMA order, the picture changes: the hybrid model significantly and consistently beats the plain LSTM on FD002. A second, architecturally unrelated CNN-LSTM baseline shows the same dataset-dependent pattern rather than a uniform result. See `paper.pdf` for the full writeup.

## Repository layout

- `paper.pdf`, `paper.tex`, `IEEEtran.cls` — the paper and its LaTeX source
- `code/` — the full experimental pipeline
  - `run_dataset.py` — main pipeline (fixed ARIMA(1,1,0), per dataset/seed)
  - `run_dataset_order_selected.py` — AIC-based per-channel ARIMA order selection variant
  - `run_cnn_lstm_baseline.py` — CNN-LSTM baseline (no ARIMA feature), per dataset/seed
  - `ablation_window.py` — window-length ablation on FD001
  - `aggregate_seeds.py`, `aggregate_ablation.py`, `aggregate_order_selected.py`, `aggregate_cnn_lstm.py` — result aggregation and significance testing
  - `analyze_cnn_lstm_significance.py` — paired Wilcoxon test, CNN-LSTM vs. plain LSTM
  - `analyze_compute_cost.py` — runtime and parameter-count overhead of ARIMA order selection
  - `make_multiseed_figures.py`, `make_ablation_figure.py`, `statistical_analysis.py` — figure generation
- `data/` — the real NASA C-MAPSS FD001-FD004 train/test data (parquet), sourced from [LucasThil's Hugging Face mirror](https://huggingface.co/datasets/LucasThil/nasa_turbofan_degradation_FD001) of the original NASA Prognostics Center of Excellence dataset
- `results/` — every raw result: per-seed predictions and metrics for all 4 datasets x 6 seeds each for the main comparison, the order-selection runs, the CNN-LSTM baseline, and the (6-seed) window-length ablation runs
- `figures/` — all paper figures (PDF + PNG)

## Reproducing a result

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn statsmodels torch matplotlib scipy pyarrow

# Main comparison for one dataset/seed:
python3 code/run_dataset.py FD001 42

# ARIMA order-selection variant:
python3 code/run_dataset_order_selected.py FD002 42

# CNN-LSTM baseline (no ARIMA feature):
python3 code/run_cnn_lstm_baseline.py FD002 42

# Window-length ablation (FD001 only):
python3 code/ablation_window.py 42

# Aggregate across seeds and run significance tests:
python3 code/aggregate_seeds.py
python3 code/aggregate_ablation.py
python3 code/aggregate_order_selected.py
python3 code/aggregate_cnn_lstm.py
python3 code/analyze_cnn_lstm_significance.py
python3 code/analyze_compute_cost.py
```

All random seeds, hyperparameters, and preprocessing choices are documented in the paper (Table I) and reproduced exactly in the code above — there is no hidden configuration.

## Disclosure

Portions of the experimental code and manuscript drafting were produced with the assistance of a generative AI system, under the direct supervision and review of the author, who verified all reported experimental results, code, and claims.

## License

Code: MIT. Data: redistributed under the original NASA C-MAPSS dataset terms (public domain, US government work).
