"""Loading processed data, station screening, and the chronological train/validation/test split."""

from pathlib import Path

import numpy as np
import pandas as pd

import config

# Columns that must be present for a row to be usable (calendar + weather at the target hour)
BASE_COLS = [
    "month", "hour", "day_of_week", "season",
    "temperature_F", "apparent_temp_F", "relative_humidity_pct",
    "wind_speed_10m_mph", "precipitation_inch", "rain_inch", "snowfall_inch",
    "cloud_cover_pct", "is_day", "direct_radiation_Wm2",
]


def load_holidays(path: Path = config.HOLIDAYS_CSV) -> set:
    """Holiday dates as 'YYYY-MM-DD' strings (empty set if the file is missing)."""
    if not Path(path).exists():
        return set()
    h = pd.read_csv(path)
    return set(pd.to_datetime(h["Date"]).dt.strftime("%Y-%m-%d").dropna())


def load_system_data(path: Path = config.SYSTEM_WEATHER_CSV) -> pd.DataFrame:
    """System-wide hourly trips + weather, sorted by time, rows with missing inputs dropped."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna(subset=BASE_COLS + config.TARGETS)
    return df.sort_values("datetime").reset_index(drop=True)


def load_station_grid(path: Path = config.SYSTEM_WEATHER_CSV, horizon: int = 1) -> pd.DataFrame:
    """Hourly grid (calendar + weather) onto which every station is merged, plus one system-level input:
    the 24-hour mean of system trips one week earlier. That lag is shift(max(168, H)), so it stays
    calendar-aligned for every horizon used here (H <= 168)."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["from_station_id"] == "ALL") & (df["to_station_id"] == "ALL")]
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    grid = df[["datetime"] + [c for c in BASE_COLS if c in df.columns]].copy()
    for t in ("outgoing", "incoming"):
        grid[f"system_lag_168h_mean_{t}"] = (
            df[f"{t}_trips"].rolling(24, min_periods=1).mean().shift(max(168, horizon)).fillna(0).values
        )
    return grid


def load_station_series(station_id, grid: pd.DataFrame, station_dir: Path = config.STATION_DIR) -> pd.DataFrame:
    """One station's trips merged onto the hourly grid; hours with no record are real zero-trip hours."""
    st = pd.read_csv(station_dir / f"station_{station_id}.csv", usecols=["datetime"] + config.TARGETS)
    st["datetime"] = pd.to_datetime(st["datetime"])
    merged = grid.merge(st, on="datetime", how="left")
    merged[config.TARGETS] = merged[config.TARGETS].fillna(0).astype(int)
    return merged.sort_values("datetime").reset_index(drop=True)


def station_is_active(station_id, station_dir: Path = config.STATION_DIR):
    """(ok, last_trip). A station is modeled only if it has real trips up to DISCONTINUED_CUTOFF;
    otherwise it closed during the study and zero-filling would fabricate its test-year demand."""
    dt = pd.to_datetime(pd.read_csv(station_dir / f"station_{station_id}.csv", usecols=["datetime"])["datetime"])
    last = dt.max() if len(dt) else pd.NaT
    return (pd.notna(last) and last >= pd.Timestamp(config.DISCONTINUED_CUTOFF)), last


def list_stations(station_dir: Path = config.STATION_DIR) -> list:
    return sorted(p.stem.replace("station_", "") for p in Path(station_dir).glob("station_*.csv"))


def train_val_test_split(df: pd.DataFrame):
    """Chronological split: train 2018-2023, validation 2024, test 2025 (copies, safe to modify)."""
    year = df["datetime"].dt.year
    train = df[(year >= config.TRAIN_YEARS[0]) & (year <= config.TRAIN_YEARS[1])].copy()
    val = df[year == config.VAL_YEAR].copy()
    test = df[year == config.TEST_YEAR].copy()
    return train, val, test


def hourly_index(start, end) -> pd.DatetimeIndex:
    """Local-time hourly index WITHOUT the spring daylight-saving hours, which never occurred. Consecutive
    entries are then exactly one elapsed hour apart, as in the row-based lags of the feature models."""
    full = pd.date_range(start, end, freq="h")
    exists = full.tz_localize("America/New_York", nonexistent="NaT", ambiguous=True).notna()
    return full[exists]


def continuous_hourly(series_df: pd.DataFrame) -> pd.DataFrame:
    """Targets on a contiguous elapsed-hour index (foundation models need a gap-free context). Genuine
    missing hours (26, all in 2018-2021) are filled with the previous value, so no filled value uses
    information from after its own hour."""
    s = series_df.set_index("datetime")[config.TARGETS]
    return s.reindex(hourly_index(s.index.min(), s.index.max())).ffill().bfill()
