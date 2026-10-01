# Evaluating Bike-Share Demand Forecasts at the Decision Horizon

Code for the paper *"Evaluating Bike-Share Demand Forecasts at the Decision Horizon: Engineered Features vs.
Time-Series Foundation Models"* (anonymous submission).

## What we did
- **Data:** Capital Bikeshare trips (Washington, D.C., 2018–2025) as hourly outgoing/incoming counts, system-wide
  and for 47 of the 50 busiest stations (3 stopped appearing in the trip records and are skipped). Hourly weather
  and archived day-ahead weather forecasts from Open-Meteo, plus U.S. federal holidays.
  Splits: train 2018–2023 / validate 2024 / test 2025, and a robustness split one year earlier.
- **Horizon-aware features:** to predict hour *T* at horizon *H*, every demand lag `shift(K)` becomes
  `shift(max(K, H))`. As a check, the same-hour-last-week baseline scores the same at every *H*.
- **Feature models:** 9 models (linear, polynomial, random forest, histogram boosting, XGBoost, LightGBM, GAM,
  two-stage zero-inflation, Poisson) + seasonal-naive + ARIMA; the best on validation is scored on test.
- **Foundation models, no fine-tuning:** TimesFM 2.0, Chronos-Bolt, and Chronos-2 with and without calendar/weather
  covariates, all scored with a rolling origin at the same *T−H* information cutoff.
- **Analyses:** 5 horizons (1–24 h); paired Wilcoxon tests across 94 station series; forecast vs. observed weather;
  second test year; feature ablation; station-volume breakdown; a flawed "legacy" TimesFM protocol for contrast.

## Repository layout
| File | What it does |
|---|---|
| `config.py` | Paths and constants (splits, horizons, seed); `BIKESHARE_TEST_YEAR=2024` selects the robustness split |
| `src/data_prep.py` | Raw trip CSVs → hourly counts (system and per station); weather and forecast-weather loaders |
| `src/data_io.py` | Loads processed data; closed-station guard; chronological split; gap-free series |
| `src/features.py` | **Horizon-aware features**; training-year station profiles; forecast-weather substitution |
| `src/models.py` | The 9 feature models, naive baseline, ARIMA; "system" and "station" hyperparameter presets |
| `src/foundation_models.py` | TimesFM, Chronos-Bolt, Chronos-2 wrappers; shared rolling-origin evaluation; legacy protocol |
| `src/metrics.py` | MAE, RMSE, R², MAPE, WMAPE (predictions clipped at 0) |
| `src/selection.py` | Model selection on the validation year |
| `scripts/01_prepare_data.py` | Step 1: build `data/processed/` from the raw files |
| `scripts/02_run_feature_models.py` | Step 2: feature models → `results/*_feature_models_h{H}.csv` (`--forecast-weather` also scores forecast weather) |
| `scripts/03_run_foundation_models.py` | Step 3: foundation models → `results/<model>_<level>.csv` |
| `scripts/04_make_tables_and_figure.py` | Step 4: writes every paper table to `tables/*.tex` and figures to `figures/` (**matplotlib**), prints all statistics (**scipy**) |
| `scripts/05_feature_ablation.py` | System-level feature ablation → `results/ablation_system.csv` |
| `results/` | Every result CSV behind the paper (`test2024/`: robustness split; `forecast_weather/`: forecast-weather runs) |

## How to run
```bash
pip install -r requirements.txt          # Python 3.12; CUDA build of PyTorch for the GPU
python scripts/01_prepare_data.py        # after placing raw data as in data/README.md (~30-60 min)
python scripts/02_run_feature_models.py --level system  --horizons 1 4 8 12 24
python scripts/02_run_feature_models.py --level station --horizons 1 4 8 12 24        # ~2 h per horizon (CPU)
python scripts/02_run_feature_models.py --level system  --horizons 1 8 24 --forecast-weather --out-dir results/forecast_weather
python scripts/02_run_feature_models.py --level station --horizons 8 --forecast-weather --out-dir results/forecast_weather
for m in timesfm chronos-bolt chronos2; do
  python scripts/03_run_foundation_models.py --model $m --level system
  python scripts/03_run_foundation_models.py --model $m --level station --batch-size 256
done
python scripts/03_run_foundation_models.py --model chronos2 --no-covariates --level system
python scripts/03_run_foundation_models.py --model chronos2 --no-covariates --level station --batch-size 256
python scripts/03_run_foundation_models.py --model chronos2 --forecast-weather --level system
python scripts/03_run_foundation_models.py --model chronos2 --forecast-weather --level station --batch-size 256
python scripts/03_run_foundation_models.py --model timesfm --level system --legacy
# robustness split (test 2024), same 47 stations as the main split
export BIKESHARE_TEST_YEAR=2024
IDS=$(python -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('results/station_feature_models_h1.csv', dtype={'station_id': str}).station_id.unique())))")
python scripts/02_run_feature_models.py --level system  --horizons 1 4 8 12 24 --out-dir results/test2024
python scripts/02_run_feature_models.py --level station --horizons 1 4 8 12 24 --stations $IDS --out-dir results/test2024
for m in timesfm chronos-bolt chronos2; do python scripts/03_run_foundation_models.py --model $m --level system --out-dir results/test2024; done
python scripts/03_run_foundation_models.py --model chronos2 --no-covariates --level system --out-dir results/test2024
unset BIKESHARE_TEST_YEAR
python scripts/05_feature_ablation.py
python scripts/04_make_tables_and_figure.py
```
Step 4 alone reproduces every table, number and figure from the CSVs in `results/`. The volume table also needs
`data/processed/stations/`. Station runs of step 3 save after every station and resume if restarted; lower
`--batch-size` if the GPU runs out of memory. Chronos-2 with covariates takes ~17 min per station on one RTX-class GPU.

## Notes
- Hyperparameters are fixed and differ by level (`PRESETS` in `src/models.py`). At system level the GAM uses the first
  20 feature columns; polynomial regression uses pairwise interactions only.
- Chronos models give median forecasts and TimesFM mean forecasts; median forecasts favour MAE, so the paper draws
  its conclusions from R².
- Weather at the target hour is observed unless `--forecast-weather` is used. The forecasts ("previous day 1") exist
  only for 2024–2025 and are complete for 2025.
- The main-split station feature-model CSVs come from the original run of this pipeline, which the refactored code
  reproduces on sampled stations and horizons; every other CSV was produced by this code.
- Foundation models ran in bfloat16 on one NVIDIA RTX 2000 Ada GPU; everything else runs on a CPU.
