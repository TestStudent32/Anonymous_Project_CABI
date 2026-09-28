"""Raw Capital Bikeshare trip files -> hourly outgoing/incoming counts (system-wide and per station),
merged with hourly weather.

Trip files come in two schemas: before 2020-04 ("Start date", "Start station number", ...) and after
("started_at", "start_station_id", ...). A trip counts as *outgoing* in the hour it starts and as
*incoming* in the hour it ends.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

CHUNK_SIZE = 200_000  # rows per chunk when scanning the large per-station file

WEATHER_RENAME = {
    "temperature_2m (°F)": "temperature_F",
    "snowfall (inch)": "snowfall_inch",
    "wind_speed_10m (mp/h)": "wind_speed_10m_mph",
    "relative_humidity_2m (%)": "relative_humidity_pct",
    "apparent_temperature (°F)": "apparent_temp_F",
    "precipitation (inch)": "precipitation_inch",
    "rain (inch)": "rain_inch",
    "cloud_cover (%)": "cloud_cover_pct",
    "wind_speed_100m (mp/h)": "wind_speed_100m_mph",
    "is_day ()": "is_day",
    "direct_radiation (W/m²)": "direct_radiation_Wm2",
}

# Column order of the processed files
COL_ORDER = [
    "datetime", "from_station_id", "to_station_id", "outgoing_trips", "incoming_trips", "total_trips",
    "year", "month", "hour", "day_of_week", "season", "date",
]


def _parse_trip_times(df: pd.DataFrame):
    """Add start/end datetime columns; return (df, schema) where schema is 'old', 'new' or None."""
    if "Start date" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["Start date"])
        df["end_datetime"] = pd.to_datetime(df["End date"])
        return df, "old"
    if "started_at" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["started_at"])
        df["end_datetime"] = pd.to_datetime(df["ended_at"])
        return df, "new"
    return df, None


def _trip_files(raw_dir: Path):
    for year_folder in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        for csv_file in sorted(year_folder.glob("*.csv")):
            yield csv_file


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


def season_code(month: pd.Series) -> np.ndarray:
    """0 = winter (Dec-Feb), 1 = spring, 2 = summer, 3 = fall."""
    return np.where(month.isin([12, 1, 2]), 0,
           np.where(month.isin([3, 4, 5]), 1,
           np.where(month.isin([6, 7, 8]), 2, 3)))


def extract_system_hourly(raw_dir: Path = config.RAW_TRIPS_DIR) -> pd.DataFrame:
    """System-wide hourly outgoing/incoming trip counts from every monthly trip file."""
    parts = []
    for csv_file in _trip_files(raw_dir):
        print(f"  {csv_file.parent.name}/{csv_file.name}")
        df, schema = _parse_trip_times(pd.read_csv(csv_file, low_memory=False))
        if schema is None:
            print("    skipped: no date columns")
            continue
        yr_s, yr_e = df["start_datetime"].dt.year, df["end_datetime"].dt.year
        df = df[(yr_s >= config.YEAR_MIN) & (yr_s <= config.YEAR_MAX) & (yr_e >= config.YEAR_MIN) & (yr_e <= config.YEAR_MAX)]
        if df.empty:
            continue
        out = df.groupby(df["start_datetime"].dt.floor("h")).size().rename("outgoing_trips")
        inc = df.groupby(df["end_datetime"].dt.floor("h")).size().rename("incoming_trips")
        hourly = pd.concat([out, inc], axis=1).fillna(0).astype(int)
        hourly.index.name = "datetime"
        parts.append(hourly.reset_index())
    # A trip can end in the next month's file window, so the same hour may appear in two files: sum them.
    result = pd.concat(parts, ignore_index=True).groupby("datetime")[["outgoing_trips", "incoming_trips"]].sum().reset_index()
    result = _add_time_columns(result)
    cols = ["datetime", "date", "year", "month", "hour", "day_of_week", "outgoing_trips", "incoming_trips"]
    return result[cols].sort_values("datetime").reset_index(drop=True)


def extract_station_hourly(raw_dir: Path = config.RAW_TRIPS_DIR) -> pd.DataFrame:
    """Per-station hourly outgoing/incoming counts. (No year filter here: out-of-range hours are dropped
    later, when each station is merged onto the system-wide hourly grid.)"""
    parts = []
    for csv_file in _trip_files(raw_dir):
        print(f"  {csv_file.parent.name}/{csv_file.name}")
        df, schema = _parse_trip_times(pd.read_csv(csv_file, low_memory=False))
        if schema is None:
            print("    skipped: no date columns")
            continue
        if schema == "old":
            s_id, s_name, e_id, e_name = "Start station number", "Start station", "End station number", "End station"
        else:
            s_id, s_name, e_id, e_name = "start_station_id", "start_station_name", "end_station_id", "end_station_name"
        keys = ["datetime", "station_number", "station_name"]
        df["start_hour"] = df["start_datetime"].dt.floor("h")
        df["end_hour"] = df["end_datetime"].dt.floor("h")
        # Trips with a missing station (e.g. dockless e-bike returns) are dropped for that side only.
        out = (df[[s_id, s_name, "start_hour"]].dropna()
               .groupby(["start_hour", s_id, s_name]).size().reset_index(name="outgoing_trips"))
        out.columns = keys + ["outgoing_trips"]
        inc = (df[[e_id, e_name, "end_hour"]].dropna()
               .groupby(["end_hour", e_id, e_name]).size().reset_index(name="incoming_trips"))
        inc.columns = keys + ["incoming_trips"]
        hourly = out.merge(inc, on=keys, how="outer").fillna(0)
        hourly[["outgoing_trips", "incoming_trips"]] = hourly[["outgoing_trips", "incoming_trips"]].astype(int)
        parts.append(hourly)
    result = pd.concat(parts, ignore_index=True).groupby(
        ["datetime", "station_number", "station_name"])[["outgoing_trips", "incoming_trips"]].sum().reset_index()
    result = _add_time_columns(result)
    cols = ["datetime", "date", "year", "month", "hour", "day_of_week", "station_number", "station_name",
            "outgoing_trips", "incoming_trips"]
    return result[cols].sort_values(["datetime", "station_number"]).reset_index(drop=True)


def load_weather(path: Path = config.WEATHER_CSV) -> pd.DataFrame:
    """Open-Meteo hourly export (3 metadata lines precede the header); columns renamed to short names."""
    df = pd.read_csv(path, skiprows=[0, 1, 2])
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["time"]).dt.floor("h")
    df = df.drop(columns=["time"]).rename(columns=WEATHER_RENAME)
    return df


def _finish(df: pd.DataFrame, weather: pd.DataFrame, station_id) -> pd.DataFrame:
    """Merge trips with weather on the hour (inner join) and add season + id columns."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    df["total_trips"] = df["outgoing_trips"] + df["incoming_trips"]
    df["from_station_id"] = station_id
    df["to_station_id"] = station_id
    merged = df.merge(weather, on="datetime", how="inner")
    merged["season"] = season_code(merged["datetime"].dt.month)
    cols = [c for c in COL_ORDER if c in merged.columns]
    return merged[cols + [c for c in merged.columns if c not in cols]]


def build_system_file(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """System-wide model input: one row per hour, station id columns set to 'ALL'."""
    return _finish(trips, weather, "ALL").sort_values("datetime").reset_index(drop=True)


def top_stations(station_hourly_csv: Path = config.STATION_HOURLY_CSV, n: int = config.TOP_N_STATIONS) -> pd.DataFrame:
    """The n stations with the most (outgoing + incoming) trips over the whole period."""
    totals, names = {}, {}
    for chunk in pd.read_csv(station_hourly_csv, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["total"] = chunk["outgoing_trips"] + chunk["incoming_trips"]
        for sn, g in chunk.groupby("station_number"):
            totals[sn] = totals.get(sn, 0) + g["total"].sum()
            names[sn] = g["station_name"].iloc[0]
    top = pd.Series(totals).sort_values(ascending=False).head(n).index.tolist()
    return pd.DataFrame({"station_number": top, "station_name": [names[s] for s in top],
                         "total_trips": [int(totals[s]) for s in top]})


def build_station_files(top: pd.DataFrame, weather: pd.DataFrame,
                        station_hourly_csv: Path = config.STATION_HOURLY_CSV, out_dir: Path = config.STATION_DIR):
    """Write one trips+weather CSV per selected station."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = top["station_number"].tolist()
    by_station = {s: [] for s in ids}
    for chunk in pd.read_csv(station_hourly_csv, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk[chunk["station_number"].isin(ids)]
        for sn, g in chunk.groupby("station_number"):
            by_station[sn].append(g)
    for sn in ids:
        if not by_station[sn]:
            print(f"  {sn}: no data")
            continue
        station = _finish(pd.concat(by_station[sn], ignore_index=True), weather, sn)
        # Rarely a station appears under two names within one hour; keep the first row, as in the paper's run.
        station = station.drop_duplicates(subset=["datetime"], keep="first").sort_values("datetime").reset_index(drop=True)
        station.to_csv(out_dir / f"station_{sn}.csv", index=False)
        print(f"  station_{sn}.csv  ({len(station)} rows)")
