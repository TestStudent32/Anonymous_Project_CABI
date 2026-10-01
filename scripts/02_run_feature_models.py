"""Step 2: train and score the feature-based models and baselines at one or more horizons.

System level : 9 feature models + seasonal-naive + ARIMA(2,0,2) on the system-wide series.
Station level: 9 feature models + seasonal-naive for each of the 47 stations still open at the end of
               the test year (3 of the top 50 closed and are skipped, with the reason printed).

Output: results/{system,station}_feature_models_h{H}.csv, one row per (station, target, model, phase)
with MAE, RMSE, R2, MAPE, WMAPE for the validation (2024) and test (2025) years.

Usage:
  python scripts/02_run_feature_models.py --level system  --horizons 1 4 8 12 24
  python scripts/02_run_feature_models.py --level station --horizons 1 --stations 31101 31200
Runtime: system ~5-10 min per horizon; station ~1.5-2.5 h per horizon for all 47 stations (CPU).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_io, features, models  # noqa: E402
from src.metrics import evaluate  # noqa: E402


def score_models(X_train, X_val, X_test, train, val, test, preset, with_arima, X_test_fw=None):
    """Yield (target, model_name, phase, metrics) for every model and both targets.
    If X_test_fw (test features with forecast weather) is given, each trained model is also scored on it
    (phase 'test_forecast_weather'): the two test sets are stacked and predicted in one call, so both
    scores come from the identical fitted model. ARIMA uses no weather and is not re-scored."""
    n_test = len(X_test)
    X_eval = X_test if X_test_fw is None else pd.concat([X_test, X_test_fw])
    for target in config.TARGETS:
        y_train, y_val, y_test = train[target], val[target], test[target]
        runs = [(name, lambda fn=fn: fn(X_train, X_val, X_eval, y_train, preset))
                for name, fn in models.FEATURE_MODELS.items()]
        runs.append((models.NAIVE_NAME, lambda: models.naive_same_hour_last_week(X_val, X_eval, y_train, target)))
        if with_arima:
            runs.append(("ARIMA", lambda: models.arima(y_train, y_val, y_test)))
        for name, run in runs:
            try:
                preds = run()
            except Exception as e:  # e.g. an optional library is missing
                print(f"    {target} / {name}: skipped ({e})")
                continue
            if preds is None:
                continue
            pred_val, pred_eval = preds
            yield target, name, "val", evaluate(y_val, pred_val)
            yield target, name, "test", evaluate(y_test, pred_eval[:n_test])
            if X_test_fw is not None and name != "ARIMA":
                yield target, name, "test_forecast_weather", evaluate(y_test, pred_eval[n_test:])


def run_system(horizon, holidays, forecast=None):
    df = features.system_features(data_io.load_system_data(), holidays, horizon)
    train, val, test = data_io.train_val_test_split(df)
    print(f"  train/val/test hours: {len(train)}/{len(val)}/{len(test)}")
    cols = features.SYSTEM_FEATURES
    X_fw = None if forecast is None else features.apply_forecast_weather(test, forecast, station=False)[cols]
    rows = [{"target": t, "model": m, "phase": ph, **met}
            for t, m, ph, met in score_models(train[cols], val[cols], test[cols], train, val, test, "system", True, X_fw)]
    return pd.DataFrame(rows)


def run_stations(horizon, holidays, station_ids, forecast=None):
    names = {}
    if config.TOP_STATIONS_CSV.exists():
        top = pd.read_csv(config.TOP_STATIONS_CSV)
        names = dict(zip(top["station_number"].astype(str), top["station_name"]))
    grid = data_io.load_station_grid(horizon=horizon)
    rows = []
    for i, sid in enumerate(station_ids, 1):
        ok, last = data_io.station_is_active(sid)
        if not ok:
            print(f"  [{i}/{len(station_ids)}] {sid}: skipped, closed (last trip {last.date() if pd.notna(last) else 'n/a'})")
            continue
        t0 = time.time()
        df = features.station_features(data_io.load_station_series(sid, grid), holidays, horizon)
        train, val, test = data_io.train_val_test_split(df)
        features.add_station_profiles(train, val, test)
        cols = features.STATION_FEATURES
        X_fw = None if forecast is None else features.apply_forecast_weather(test, forecast, station=True)[cols]
        for t, m, ph, met in score_models(train[cols], val[cols], test[cols], train, val, test, "station", False, X_fw):
            rows.append({"station_id": sid, "station_name": names.get(sid, ""), "target": t, "model": m, "phase": ph, **met})
        print(f"  [{i}/{len(station_ids)}] {sid}: done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["system", "station"], required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--stations", nargs="*", default=None, help="Station IDs (default: all station files)")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--forecast-weather", action="store_true",
                        help="Also score every model with archived day-ahead weather forecasts at the target hour")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # convergence warnings (e.g. Poisson) are expected and reported in the paper
    args.out_dir.mkdir(parents=True, exist_ok=True)
    holidays = data_io.load_holidays()
    forecast = None
    if args.forecast_weather:
        from src.data_prep import load_forecast_weather
        forecast = load_forecast_weather()

    for h in args.horizons:
        print(f"== {args.level} level, horizon H={h} ==")
        if args.level == "system":
            res = run_system(h, holidays, forecast)
        else:
            res = run_stations(h, holidays, args.stations or data_io.list_stations(), forecast)
        out = args.out_dir / f"{args.level}_feature_models_h{h}.csv"
        res.to_csv(out, index=False)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
