"""Step 3: zero-shot TimesFM on the same test hours as the feature models.

  --level system              rolling-origin protocol, H in {1,4,8,12,24}  -> results/timesfm_system.csv
  --level system --legacy     the flawed legacy protocol (freq=1 and freq=0) -> results/timesfm_system_legacy.csv
  --level station             rolling-origin for each of the 47 stations    -> results/timesfm_station.csv
                              (checkpointed after every station; re-running resumes)

Usage:  python scripts/03_run_timesfm.py --level station --batch-size 64
Runtime (one RTX-class GPU, bfloat16): system ~1 min; stations ~45 min. CPU works but is much slower.
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
from src.metrics import evaluate  # noqa: E402
from src.timesfm_forecaster import TimesFM, legacy_recursive, rolling_origin  # noqa: E402


def system_test_times(holidays) -> pd.DatetimeIndex:
    """Exactly the test hours scored for the system-level feature models."""
    feat = features.system_features(data_io.load_system_data(), holidays, horizon=1)
    return pd.DatetimeIndex(data_io.train_val_test_split(feat)[2]["datetime"])


def run_system(tfm, holidays, batch_size):
    series = data_io.continuous_hourly(data_io.load_system_data())
    test_times = system_test_times(holidays)
    rows = []
    for target in config.TARGETS:
        preds = rolling_origin(tfm, series[target], test_times, config.HORIZONS, batch_size)
        y_true = series.loc[test_times, target].values
        for h in config.HORIZONS:
            rows.append({"target": target, "horizon": h, **evaluate(y_true, preds[h])})
            print(f"  H={h:>2} {target}: R2={rows[-1]['R2']:.3f}")
    return pd.DataFrame(rows)


def run_system_legacy(tfm, holidays):
    series = data_io.continuous_hourly(data_io.load_system_data())
    test_times = system_test_times(holidays)
    hist_end = pd.Timestamp(f"{config.VAL_YEAR}-12-31 23:00:00")
    test_range = pd.date_range(hist_end + pd.Timedelta(hours=1), test_times.max(), freq="h")
    rows = []
    for target in config.TARGETS:
        for freq in (config.TIMESFM_MEDIUM_FREQ, config.TIMESFM_HIGH_FREQ):
            pred = legacy_recursive(tfm, series.loc[:hist_end, target].values, len(test_range), freq)
            pred = pd.Series(pred, index=test_range).reindex(test_times).values
            rows.append({"target": target, "protocol": f"legacy_recursive_freq{freq}",
                         **evaluate(series.loc[test_times, target].values, pred)})
            print(f"  legacy freq={freq} {target}: R2={rows[-1]['R2']:.3f}")
    return pd.DataFrame(rows)


def run_stations(tfm, holidays, station_ids, batch_size, out_path):
    grid = data_io.load_station_grid(horizon=1)
    done = pd.read_csv(out_path, dtype={"station_id": str}) if out_path.exists() else pd.DataFrame()
    finished = set(done["station_id"]) if len(done) else set()
    rows = done.to_dict("records")
    for i, sid in enumerate(station_ids, 1):
        ok, _ = data_io.station_is_active(sid)
        if not ok or sid in finished:
            continue
        t0 = time.time()
        raw = data_io.load_station_series(sid, grid)
        feat = features.station_features(raw, holidays, horizon=1)
        test_times = pd.DatetimeIndex(feat.loc[feat["datetime"].dt.year == config.TEST_YEAR, "datetime"])
        series = data_io.continuous_hourly(raw)
        for target in config.TARGETS:
            preds = rolling_origin(tfm, series[target], test_times, config.HORIZONS, batch_size)
            y_true = series.loc[test_times, target].values
            for h in config.HORIZONS:
                rows.append({"station_id": sid, "target": target, "horizon": h, **evaluate(y_true, preds[h])})
        pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint
        print(f"  [{i}/{len(station_ids)}] {sid}: done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["system", "station"], required=True)
    parser.add_argument("--legacy", action="store_true", help="System level: run the flawed legacy protocol")
    parser.add_argument("--stations", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    holidays = data_io.load_holidays()
    tfm = TimesFM()
    print(f"TimesFM on {tfm.device} ({tfm.dtype}), context {tfm.context_length} h")

    if args.level == "system" and args.legacy:
        out, res = args.out_dir / "timesfm_system_legacy.csv", run_system_legacy(tfm, holidays)
    elif args.level == "system":
        out, res = args.out_dir / "timesfm_system.csv", run_system(tfm, holidays, args.batch_size)
    else:
        out = args.out_dir / "timesfm_station.csv"
        res = run_stations(tfm, holidays, args.stations or data_io.list_stations(), args.batch_size, out)
    res.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
