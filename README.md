# Nowcasting or Forecasting? Horizon-Aware Evaluation of Bike-Share Demand Models

Code for the paper *"Nowcasting or Forecasting? Horizon-Aware Evaluation of Feature-Based and Zero-Shot
Foundation Models for Bike-Share Demand"* (anonymous submission).

## What we did
- **Data:** Capital Bikeshare trips (Washington, D.C., 2018–2025), turned into hourly outgoing and
  incoming counts, system-wide and for the 50 busiest stations. 3 of these stations closed during the
  study and are skipped, leaving 47. Hourly weather comes from Open-Meteo, plus U.S. federal holidays.
  Split: train 2018–2023, validation 2024, test 2025.
- **Horizon-aware features:** to predict hour *T* at horizon *H*, every demand lag `shift(K)` becomes
  `shift(max(K, H))`, so no input comes from after *T−H*. As a check, the same-hour-last-week baseline
  scores exactly the same at every *H*.
- **Models:** 9 feature-based models (linear, polynomial, random forest, histogram gradient boosting,
  XGBoost, LightGBM, GAM, a two-stage zero-inflation model, Poisson), a seasonal-naive baseline, and
  ARIMA at system level only. Per station, the model with the best validation R² is chosen.
- **Zero-shot TimesFM** (`timesfm-2.0-500m`, no fine-tuning, no features): a rolling-origin evaluation
  with the same *T−H* cutoff. A flawed "legacy" protocol is also reproduced to show how the evaluation
  setup changes the conclusion.
- **Horizons** H ∈ {1, 4, 8, 12, 24} h. Metrics: R², MAE, plus RMSE/MAPE/WMAPE. Station-level
  comparisons use a paired Wilcoxon signed-rank test across the 94 station–direction series.

## Repository layout
| File | What it does |
|---|---|
| `config.py` | All paths and constants (split years, horizons, seed, TimesFM model id) |
| `src/data_prep.py` | Raw trip CSVs → hourly counts (system and per station); merges weather; selects the top 50 stations |
| `src/data_io.py` | Loads processed data; closed-station guard; chronological split; gap-free series for TimesFM |
| `src/features.py` | **Horizon-aware features** (`shift(max(K,H))`); station profiles fit on training years only |
| `src/models.py` | The 9 feature models, the naive baseline and ARIMA; "system" and "station" hyperparameter presets |
| `src/metrics.py` | MAE, RMSE, R², MAPE, WMAPE (predictions clipped at 0) |
| `src/selection.py` | Per-station model selection on the validation year |
| `src/timesfm_forecaster.py` | TimesFM wrapper; rolling-origin and legacy protocols |
| `scripts/01_prepare_data.py` | Step 1: build `data/processed/` from the raw files |
| `scripts/02_run_feature_models.py` | Step 2: feature models and baselines → `results/*_feature_models_h{H}.csv` |
| `scripts/03_run_timesfm.py` | Step 3: TimesFM → `results/timesfm_*.csv` |
| `scripts/04_make_tables_and_figure.py` | Step 4: prints Tables 1–2 and the statistics; draws Fig. 1 (**matplotlib**) → `figures/horizon.png` |
| `results/` | The result CSVs behind every number in the paper |

## How to run
```bash
pip install -r requirements.txt           # Python 3.12; a CUDA build of PyTorch for the GPU
# put the raw files in data/raw/ as described in data/README.md, then:
python scripts/01_prepare_data.py                                   # ~30-60 min
python scripts/02_run_feature_models.py --level system  --horizons 1 4 8 12 24
python scripts/02_run_feature_models.py --level station --horizons 1 4 8 12 24   # ~2 h per horizon (CPU)
python scripts/03_run_timesfm.py --level system
python scripts/03_run_timesfm.py --level system --legacy
python scripts/03_run_timesfm.py --level station --batch-size 64      # ~45 min on one GPU; use 32 if out of memory
python scripts/04_make_tables_and_figure.py
```
Step 4 alone reproduces every table, number and the figure from the CSVs already in `results/`.

## Notes
- Hyperparameters differ by level (see `PRESETS` in `src/models.py`): the station preset uses
  absolute-error boosting with more, deeper trees. At system level the GAM uses the first 20 feature
  columns, and polynomial regression uses pairwise interactions only.
- TimesFM ran in bfloat16 on one NVIDIA RTX 2000 Ada GPU. Everything else runs on a CPU.
- Weather at the target hour is taken from historical data (a perfect-forecast assumption), as stated
  in the paper.
- The station feature-model results in `results/` come from the original run of this pipeline. The
  refactored code reproduces them (checked on sampled stations and horizons); every other CSV was
  regenerated with this code.
