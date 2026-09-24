# Reliable Evaluation of ARIMA-LSTM Hybrids for Remaining Useful Life Prediction

Code, data, and per-seed results for *"Reliable Evaluation of ARIMA-LSTM Hybrids for Remaining Useful Life Prediction: A Multi-Seed, Specification-Sensitive Study on NASA C-MAPSS"*, accepted at ICAIET-2027 (Machine Learning and Data Science track).

## What this is

We evaluate, rather than propose, ARIMA trend augmentation for LSTM-based remaining useful life (RUL) prediction on all four NASA C-MAPSS sub-datasets. Three ARIMA specifications are compared against a plain LSTM (and Random Forest and CNN-LSTM baselines): a fixed order, an AIC-selected order (both fit on each unit's full series, hence non-causal), and a fully causal expanding-window construction that never uses observations after the predicted cycle.

Main findings, from the six-seed study reported in the paper:

- No specification yields a corrected-significant improvement of the hybrid over the plain LSTM on any dataset.
- Conclusions moved with seed count: a significant three-seed result on FD001 disappeared at six seeds.
- A free, fitting-free causal moving average matches or beats the far more expensive causal ARIMA feature (about 2,000 times cheaper to construct).
- The two engine-level significant results in the paper (fixed-order hybrid worse than plain on FD004; moving average better than causal ARIMA on FD002) are nominal only under a two-level seeds-by-engines test (`code/hier_bootstrap.py`).

An earlier version of this README claimed the hybrid significantly beats the plain LSTM on FD002. That claim predates multiple-comparison correction, does not hold, and has been removed.

## Repository layout

- `paper_icaiet_camera_ready.pdf/.tex` : the 6-page accepted paper. `paper.pdf/.tex` : the full-length technical report (all ablations and negative results).
- `code/`
  - `run_dataset.py` (fixed-order hybrid, plain LSTM, Random Forest), `run_dataset_order_selected.py` (AIC-selected order), `run_dataset_causal.py` (causal expanding-window ARIMA), `run_trend_controls.py` (moving-average and exponential-smoothing features), `run_capacity_controls.py` (parameter-matched and duplicated-channel plain LSTMs), `run_cnn_lstm_baseline.py`, `run_gated_fusion.py`, `ablation_window.py`
  - `feature_cache.py` : dataset-level cache of ARIMA features, guarded by a fingerprint of the normalised data
  - `run_extra_seeds.py` : runs every variant for additional seeds (resumable)
  - `equiv_seedbudget.py`, `hier_bootstrap.py`, `analysis_extended.py` : equivalence tests, two-level bootstrap, seed-budget analysis
  - `timing_feature_construction.py` : controlled CPU-time comparison of feature constructions
  - `aggregate_*.py`, `analyze_*.py`, `make_*_figures.py`, `statistical_analysis.py` : aggregation and figures
- `data/` : NASA C-MAPSS FD001-FD004 (parquet), from [LucasThil's Hugging Face mirror](https://huggingface.co/datasets/LucasThil/nasa_turbofan_degradation_FD001) of the NASA Prognostics Center of Excellence data
- `results/` : per-seed metrics and per-engine predictions for every variant
- `requirements.txt`, `reproduce.sh` : exact package versions and the full reproduction sequence

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # exact versions used for every result
python3 code/run_dataset.py FD001 42     # one dataset and seed
./reproduce.sh                            # everything (hours)
```

With the pinned versions on CPU, reruns reproduce the archived RMSE values exactly (verified for FD001 seeds 42 and 123, including with a different thread count). The causal ARIMA feature costs about 222 ms of CPU per series against 0.10 ms for the moving average, so features are computed once per dataset and reused across seeds; reuse is guarded by a SHA-256 fingerprint of the normalised series and is rejected if the data differ.

## Known limitations

- Six seeds is moderate power; seed variation dominates the uncertainty of every comparison (see `results/hier_bootstrap.json`).
- C-MAPSS is simulated data; generalisation to real industrial streams is untested.
- The original order-selected sweep did not archive per-engine predictions; they are being regenerated (`code/run_extra_seeds.py`).

## Disclosure

Portions of the experimental code and manuscript drafting were produced with the assistance of a generative AI system, under the direct supervision and review of the author, who verified all reported experimental results, code, and claims.

## License

Code: MIT. Data: redistributed under the original NASA C-MAPSS dataset terms (public domain, US government work).
