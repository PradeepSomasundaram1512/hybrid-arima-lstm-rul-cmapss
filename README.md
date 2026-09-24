# Reliable Evaluation of ARIMA-LSTM Hybrids for Remaining Useful Life Prediction

Code, data, and per-seed results for *"Reliable Evaluation of ARIMA-LSTM Hybrids for Remaining Useful Life Prediction: A Multi-Seed, Specification-Sensitive Study on NASA C-MAPSS"*, accepted at ICAIET-2027 (Machine Learning and Data Science track).

## What this is

We evaluate, rather than propose, ARIMA trend augmentation for LSTM-based remaining useful life (RUL) prediction on all four NASA C-MAPSS sub-datasets, over **20 seeds**. A plain LSTM is compared with hybrids using three ARIMA specifications (fixed order, AIC-selected order, and a causal expanding-window construction), two cheap causal smoothers (moving average, exponential smoothing), and two capacity controls (a parameter-matched plain LSTM and a duplicated-input plain LSTM).

Main findings (20 seeds):

- **Effects depend on the dataset and the test.** On FD002 trend features lower RMSE at the seed level, most with (non-causal) AIC-selected ARIMA (-1.00 cycles, Holm p = 0.0004) and a moving average (-0.69), but none of the FD002 effects survives the two-level test; on FD001 and FD003 they do not help.
- **Input width likely contributes to the FD004 penalty.** Duplicating the input with no new information gives a penalty of similar size (+0.50 RMSE), so the penalty need not implicate the ARIMA trend; this does not show that width is the only cause.
- **Causal construction did not cost accuracy.** Causal and fixed-order hybrids did not differ beyond a prespecified ±1-cycle margin (Holm-adjusted TOST), at about 44 times the feature-construction cost.
- **A moving average had lower RMSE than causal ARIMA** on three of four datasets at the seed level (none survives the two-level test), at about 1/2000 of the feature cost.
- **Few-seed conclusions are fragile.** The original six seeds reversed the sign of the mean effect in 6 of 20 comparisons. Seed-level tests find 15 of 48 comparison cells significant, but a stricter two-level test (seeds and test engines as crossed factors) finds only 2.

Earlier versions of this repository stated that the hybrid "significantly and consistently beats the plain LSTM on FD002" and, later, that no hybrid beats the plain LSTM. Neither statement is supported once 20 seeds and both test levels are used; the statements above replace them.

## Repository layout

- `paper_icaiet_camera_ready_20seed.pdf/.tex` : the 6-page paper (20-seed evidence, recommended final version). `paper.pdf/.tex` : the same paper plus an appendix (cost, convergence, six-seed ablation and gated fusion). `paper_icaiet_camera_ready.pdf/.tex` : the earlier six-seed version that was reviewed.
- `archive_old_versions/` : superseded early drafts (short and blind-review versions), kept for the record only.
- `code/`
  - `run_dataset.py` (plain LSTM, Random Forest, fixed-order hybrid), `run_dataset_order_selected.py` (AIC-selected order), `run_dataset_causal.py` (causal expanding-window ARIMA), `run_trend_controls.py` (moving average, exponential smoothing), `run_capacity_controls.py` (parameter-matched and duplicated-input plain LSTMs), `run_cnn_lstm_baseline.py`, `run_gated_fusion.py`, `ablation_window.py`
  - `feature_cache.py` : dataset-level cache of ARIMA features, guarded by a SHA-256 fingerprint of the normalised data
  - `run_extra_seeds.py` : runs every variant for the 14 additional seeds (1 to 15 except 7, fixed in advance) and restores the AIC-selected per-engine predictions; resumable
  - `analysis_extended.py` (all comparisons, seed-level and two-level tests, equivalence, seed budget, power), `variance_shares.py` (48 cells, the 12 comparisons of Table II), `hier_bootstrap.py`, `equiv_seedbudget.py`, `timing_feature_construction.py`
  - `make_extended_figures.py`, `build_paper_tables.py` : figures and LaTeX tables generated from the results
- `data/` : NASA C-MAPSS FD001-FD004 (parquet), from [LucasThil's Hugging Face mirror](https://huggingface.co/datasets/LucasThil/nasa_turbofan_degradation_FD001) of the NASA Prognostics Center of Excellence data
- `results/` : per-seed metrics and per-engine predictions for every variant; `results/extended_analysis.json` holds every statistic in the paper
- `requirements.txt`, `reproduce.sh` : exact package versions and the full reproduction sequence

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # exact versions used for every result
python3 code/run_dataset.py FD001 42     # one dataset and seed
./reproduce.sh                            # everything (hours)
python3 code/analysis_extended.py         # statistics on the archived predictions
```

With the pinned versions on CPU, reruns reproduce archived results essentially exactly: the main pipeline matched to four decimals on FD001 (seeds 42 and 123, also with a different thread count), and all 24 AIC-selected reruns matched their earlier RMSEs to within 4e-15. The causal ARIMA feature costs about 222 ms of CPU per series against 0.10 ms for the moving average, so features are computed once per dataset and reused across seeds; each reuse is verified against a fingerprint of the normalised series and rejected if the data differ.

## Known limitations

- Twenty seeds is moderate power for effects of a fraction of a cycle (about 10 to 67 seeds would be needed to detect a half-cycle gap).
- One simulated benchmark, one recurrent family and a fixed test set; generalisation to real industrial data is untested.
- AIC-selected order is fit on each unit's full series (non-causal); no causal AIC selection was run.
- CNN-LSTM, gated fusion and the window-length ablation were run on six seeds only.
- About 6.5% to 7.0% of ARIMA fits did not fully converge; they are kept.

## Disclosure

Portions of the experimental code and manuscript drafting were produced with the assistance of a generative AI system, under the direct supervision and review of the author, who verified all reported experimental results, code, and claims.

## License

Code: MIT. Data: redistributed under the original NASA C-MAPSS dataset terms (public domain, US government work).
