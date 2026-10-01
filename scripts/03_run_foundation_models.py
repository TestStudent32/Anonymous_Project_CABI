"""Step 3: pretrained foundation models (no fine-tuning), scored on the same test hours as the feature models.

  --model timesfm        demand history only                         -> results/timesfm_<level>.csv
  --model chronos-bolt   demand history only                         -> results/chronos_bolt_<level>.csv
  --model chronos2       + calendar and weather covariates           -> results/chronos2_<level>.csv
  --model chronos2 --no-covariates                                   -> results/chronos2_nocov_<level>.csv
  --model timesfm --level system --legacy   flawed legacy protocol   -> results/timesfm_system_legacy.csv

Station runs are checkpointed after every station; re-running resumes.
Usage:   python scripts/03_run_foundation_models.py --model chronos2 --level station --batch-size 64
Runtime (one RTX-class GPU, bfloat16): system ~1-3 min per model; stations ~45-90 min per model.
If the GPU runs out of memory, lower --batch-size.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_io, features  # noqa: E402
from src.foundation_models import Chronos2, ChronosBolt, TimesFM, legacy_recursive, rolling_origin  # noqa: E402
from src.metrics import evaluate  # noqa: E402

MODELS = {"timesfm": TimesFM, "chronos-bolt": ChronosBolt, "chronos2": Chronos2}
# Known-in-advance covariates for Chronos-2: calendar + weather at the forecast hours
COVARIATES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday_or_weekend"] + features._WEATHER_NOW


def covariate_frame(df: pd.DataFrame, holidays: set, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar + weather covariates on the gap-free hourly index used for the demand series."""
    cal = features.add_calendar_features(df.copy(), holidays).set_index("datetime")[COVARIATES]
    return cal.reindex(index).interpolate(limit_direction="both")


def evaluate_series(model, df, test_times, holidays, use_cov, batch_size, keys, forecast=None):
    """Rolling-origin forecasts for both targets of one series; returns result rows. With `forecast`,
    the future weather covariates come from archived day-ahead forecasts (past ones stay observed)."""
    series = data_io.continuous_hourly(df)
    cov = covariate_frame(df, holidays, series.index) if use_cov else None
    fcov = None
    if use_cov and forecast is not None:
        fdf = features.apply_forecast_weather(df, forecast, station=False)
        fcov = covariate_frame(fdf, holidays, series.index)
    rows = []
    for target in config.TARGETS:
        preds = rolling_origin(model, series[target], test_times, config.HORIZONS, cov, batch_size, fcov)
        y_true = series.loc[test_times, target].values
        for h in config.HORIZONS:
            rows.append({**keys, "target": target, "horizon": h, **evaluate(y_true, preds[h])})
    return rows


def run_legacy(model, holidays):
    series = data_io.continuous_hourly(data_io.load_system_data())
    feat = features.system_features(data_io.load_system_data(), holidays, horizon=1)
    test_times = pd.DatetimeIndex(data_io.train_val_test_split(feat)[2]["datetime"])
    hist_end = pd.Timestamp(f"{config.VAL_YEAR}-12-31 23:00:00")
    test_range = pd.date_range(hist_end + pd.Timedelta(hours=1), test_times.max(), freq="h")
    rows = []
    for target in config.TARGETS:
        for freq in (config.TIMESFM_MEDIUM_FREQ, config.TIMESFM_HIGH_FREQ):
            pred = legacy_recursive(model, series.loc[:hist_end, target].values, len(test_range), freq)
            pred = pd.Series(pred, index=test_range).reindex(test_times).values
            rows.append({"target": target, "protocol": f"legacy_recursive_freq{freq}",
                         **evaluate(series.loc[test_times, target].values, pred)})
            print(f"  legacy freq={freq} {target}: R2={rows[-1]['R2']:.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--level", choices=["system", "station"], required=True)
    parser.add_argument("--no-covariates", action="store_true", help="Chronos-2 without covariates")
    parser.add_argument("--legacy", action="store_true", help="TimesFM, system level: flawed legacy protocol")
    parser.add_argument("--forecast-weather", action="store_true",
                        help="Chronos-2: future weather covariates from archived day-ahead forecasts")
    parser.add_argument("--stations", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    holidays = data_io.load_holidays()
    model = MODELS[args.model]()
    use_cov = model.uses_covariates and not args.no_covariates
    slug = args.model.replace("-", "_") + ("_nocov" if model.uses_covariates and not use_cov else "")
    forecast = None
    if args.forecast_weather and use_cov:
        from src.data_prep import load_forecast_weather
        forecast = load_forecast_weather()
        slug += "_fcstweather"
    print(f"{model.name} on {model.device} ({model.dtype}), covariates={use_cov}")

    if args.legacy:
        out = args.out_dir / "timesfm_system_legacy.csv"
        run_legacy(model, holidays).to_csv(out, index=False)
        print(f"-> {out}")
        return

    out = args.out_dir / f"{slug}_{args.level}.csv"
    if args.level == "system":
        df = data_io.load_system_data()
        feat = features.system_features(df, holidays, horizon=1)
        test_times = pd.DatetimeIndex(data_io.train_val_test_split(feat)[2]["datetime"])
        rows = evaluate_series(model, df, test_times, holidays, use_cov, args.batch_size, {}, forecast)
        for r in rows:
            print(f"  H={r['horizon']:>2} {r['target']}: R2={r['R2']:.3f}")
    else:
        grid = data_io.load_station_grid(horizon=1)
        done = pd.read_csv(out, dtype={"station_id": str}) if out.exists() else pd.DataFrame()
        finished = set(done["station_id"]) if len(done) else set()
        rows = done.to_dict("records")
        station_ids = args.stations or data_io.list_stations()
        for i, sid in enumerate(station_ids, 1):
            ok, _ = data_io.station_is_active(sid)
            if not ok or sid in finished:
                continue
            t0 = time.time()
            raw = data_io.load_station_series(sid, grid)
            feat = features.station_features(raw, holidays, horizon=1)
            test_times = pd.DatetimeIndex(feat.loc[feat["datetime"].dt.year == config.TEST_YEAR, "datetime"])
            rows += evaluate_series(model, raw, test_times, holidays, use_cov, args.batch_size,
                                    {"station_id": sid}, forecast)
            pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint
            print(f"  [{i}/{len(station_ids)}] {sid}: done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
