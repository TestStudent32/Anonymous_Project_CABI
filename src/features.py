"""Horizon-aware feature construction.

Rule: to predict hour T with horizon H, a demand lag with nominal offset K hours becomes
    shift(K) -> shift(max(K, H)).
Calendar-aligned lags (24 h = same hour yesterday, 168 h = same hour last week, ...) keep their exact
meaning whenever K >= H; shorter lags move back to T-H, the freshest value known at decision time.
Rolling and exponentially weighted features are built on the series already shifted by max(1, H).
Weather at the target hour is NOT shifted (perfect-weather-forecast assumption, stated in the paper).

Check: the same-hour-last-week naive forecast uses lag_168h, so its accuracy must be identical at
every H <= 168. It is (R^2 = 0.782 at all five horizons).

NOTE: the feature lists below are kept in the exact order used for the paper; column order affects
tree models (random feature subsets) and the system-level GAM (first 20 columns).
"""

import numpy as np
import pandas as pd

import config

WEATHER_LAG_VARS = ["temperature_F", "apparent_temp_F", "wind_speed_10m_mph",
                    "precipitation_inch", "rain_inch", "snowfall_inch", "cloud_cover_pct"]
WEATHER_LAGS = (1, 3, 6, 12)
_WEATHER_LAG_COLS = [f"{v}_lag{h}h" for v in WEATHER_LAG_VARS for h in WEATHER_LAGS]

_CALENDAR = ["season_winter", "season_spring", "season_summer", "season_fall",
             "hour_sin", "hour_cos", "dow_sin", "dow_cos",
             "is_holiday_or_weekend", "active_hours_flag", "is_covid_period"]
_WEATHER_NOW = ["temperature_F", "apparent_temp_F", "relative_humidity_pct", "wind_speed_10m_mph",
                "precipitation_inch", "rain_inch", "snowfall_inch", "cloud_cover_pct", "direct_radiation_Wm2"]

SYSTEM_FEATURES = _CALENDAR + [
    "lag_1h_outgoing", "lag_24h_outgoing", "lag_168h_outgoing", "lag_336h_mean_outgoing",
    "lag_1h_incoming", "lag_24h_incoming", "lag_168h_incoming", "lag_336h_mean_incoming",
    "rolling_7d_outgoing", "rolling_24h_outgoing",
    "rolling_7d_incoming", "rolling_24h_incoming",
    "ewm_24h_outgoing", "ewm_24h_incoming",
    "ewm_168h_outgoing", "ewm_168h_incoming",
] + _WEATHER_NOW + _WEATHER_LAG_COLS

STATION_FEATURES = _CALENDAR + [
    "lag_1h_outgoing", "lag_24h_outgoing", "lag_168h_outgoing", "lag_336h_mean_outgoing",
    "lag_336h_outgoing", "lag_504h_outgoing", "lag_672h_outgoing",
    "lag_1h_incoming", "lag_24h_incoming", "lag_168h_incoming", "lag_336h_mean_incoming",
    "lag_336h_incoming", "lag_504h_incoming", "lag_672h_incoming",
    "lag_4w_same_hour_outgoing", "lag_4w_same_hour_incoming",
    "rolling_7d_outgoing", "rolling_24h_outgoing", "rolling_7d_median_outgoing",
    "rolling_7d_incoming", "rolling_24h_incoming", "rolling_7d_median_incoming",
    "ewm_24h_outgoing", "ewm_24h_incoming",
    "ewm_168h_outgoing", "ewm_168h_incoming",
    "system_lag_168h_mean_outgoing", "system_lag_168h_mean_incoming",
    "station_share_168h_outgoing", "station_share_168h_incoming",
    "hist_hour_of_week_outgoing", "hist_hour_of_week_incoming",
    "hist_month_hour_outgoing", "hist_month_hour_incoming",
    "station_mean_outgoing", "station_mean_incoming",
    "ratio_lag168_hist_outgoing", "ratio_lag168_hist_incoming",
] + _WEATHER_NOW + [
    "bad_weather", "feels_like_diff", "temp_comfort",
    "rain_x_rush", "rain_x_leisure", "temp_x_summer", "temp_x_winter", "snow_x_weekday",
] + _WEATHER_LAG_COLS

# Features fit on the training years only (filled by add_station_profiles after the split)
PROFILE_FEATURES = ["hist_hour_of_week_outgoing", "hist_hour_of_week_incoming",
                    "hist_month_hour_outgoing", "hist_month_hour_incoming",
                    "station_mean_outgoing", "station_mean_incoming",
                    "ratio_lag168_hist_outgoing", "ratio_lag168_hist_incoming"]


def add_calendar_features(df: pd.DataFrame, holidays: set) -> pd.DataFrame:
    """Holiday/weekend and COVID flags, cyclic hour and day-of-week, active hours, one-hot season."""
    dates = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    is_holiday = dates.isin(holidays)
    df["is_covid_period"] = ((df["datetime"] >= config.COVID_START) & (df["datetime"] <= config.COVID_END)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_holiday_or_weekend"] = (is_holiday | (df["day_of_week"] >= 5)).astype(int)
    df["active_hours_flag"] = ((df["hour"] >= 6) & (df["hour"] <= 22)).astype(int)
    for code, name in enumerate(["winter", "spring", "summer", "fall"]):
        df[f"season_{name}"] = (df["season"] == code).astype(int)
    return df


def _add_demand_lags(df: pd.DataFrame, horizon: int, station: bool) -> None:
    """Target lags, rolling and EWM features for both directions, all under shift(max(K, H))."""
    hs = lambda k: max(k, horizon)  # noqa: E731
    for t in ("outgoing", "incoming"):
        y = df[f"{t}_trips"]
        df[f"lag_1h_{t}"] = y.shift(hs(1))
        df[f"lag_24h_{t}"] = y.shift(hs(24))
        df[f"lag_168h_{t}"] = y.shift(hs(168))
        df[f"lag_336h_mean_{t}"] = y.rolling(24, min_periods=1).mean().shift(hs(336))
        if station:  # longer same-hour lags for the noisier station series
            df[f"lag_336h_{t}"] = y.shift(hs(336))
            df[f"lag_504h_{t}"] = y.shift(hs(504))
            df[f"lag_672h_{t}"] = y.shift(hs(672))
            df[f"lag_4w_same_hour_{t}"] = y.rolling(168, min_periods=1).mean().shift(hs(672))
    for t in ("outgoing", "incoming"):
        y_known = df[f"{t}_trips"].shift(hs(1))  # latest value available at decision time
        df[f"rolling_7d_{t}"] = y_known.rolling(168, min_periods=1).mean()
        df[f"rolling_24h_{t}"] = y_known.rolling(24, min_periods=1).mean()
        if station:
            df[f"rolling_7d_median_{t}"] = y_known.rolling(168, min_periods=1).median()
    for t in ("outgoing", "incoming"):
        y_known = df[f"{t}_trips"].shift(hs(1))
        df[f"ewm_24h_{t}"] = y_known.ewm(span=24, adjust=False).mean()
        df[f"ewm_168h_{t}"] = y_known.ewm(span=168, adjust=False).mean()


def _add_weather_lags(df: pd.DataFrame, horizon: int) -> None:
    """Weather 1/3/6/12 h before the target, also subject to shift(max(K, H))."""
    for v in WEATHER_LAG_VARS:
        for h in WEATHER_LAGS:
            df[f"{v}_lag{h}h"] = df[v].shift(max(h, horizon))


def system_features(df: pd.DataFrame, holidays: set, horizon: int = 1) -> pd.DataFrame:
    """Feature table for the system-wide series. The first max(336, H) hours are dropped (lag warm-up)."""
    df = add_calendar_features(df.copy(), holidays)
    _add_demand_lags(df, horizon, station=False)
    _add_weather_lags(df, horizon)
    df = df.iloc[max(336, horizon):].reset_index(drop=True)
    fill = [c for c in SYSTEM_FEATURES if c.startswith(("lag_", "rolling_", "ewm_")) or c in _WEATHER_LAG_COLS]
    df[fill] = df[fill].fillna(0)
    return df.dropna(subset=SYSTEM_FEATURES + config.TARGETS)


def station_features(df: pd.DataFrame, holidays: set, horizon: int = 1) -> pd.DataFrame:
    """Feature table for one station (input: output of data_io.load_station_series).
    The first max(672, H) hours are dropped (4-week lag warm-up). Weather lags are computed after that
    cut, so their first few rows are 0-filled (only affects January 2018, i.e. training data)."""
    df = add_calendar_features(df.copy(), holidays)
    _add_demand_lags(df, horizon, station=True)
    df = df.iloc[max(672, horizon):].reset_index(drop=True)
    _add_weather_lags(df, horizon)
    fill = [c for c in STATION_FEATURES if c.startswith(("lag_", "rolling_", "ewm_")) or c in _WEATHER_LAG_COLS]
    df[fill] = df[fill].fillna(0)
    # Station's share of system volume one week earlier (both terms are lagged, so horizon-safe)
    for t in ("outgoing", "incoming"):
        sys_vol = df[f"system_lag_168h_mean_{t}"] + 1e-6
        df[f"station_share_168h_{t}"] = np.where(sys_vol > 1e-6, df[f"lag_168h_{t}"] / sys_vol, 0)
    # Weather composites and interactions (weather at the target hour)
    rush = ((df["hour"] >= 7) & (df["hour"] <= 9)) | ((df["hour"] >= 16) & (df["hour"] <= 18))
    rain = df["rain_inch"].fillna(0)
    df["bad_weather"] = ((df["rain_inch"] > 0.1) | (df["snowfall_inch"] > 0) | (df["wind_speed_10m_mph"] > 20)).astype(int)
    df["feels_like_diff"] = (df["apparent_temp_F"] - df["temperature_F"]).fillna(0)
    df["temp_comfort"] = ((df["temperature_F"] >= 55) & (df["temperature_F"] <= 80)).astype(int)
    df["rain_x_rush"] = (rain * rush).astype(float)
    df["rain_x_leisure"] = (rain * (~rush)).astype(float)
    df["temp_x_summer"] = (df["temperature_F"].fillna(0) * df["season_summer"]).astype(float)
    df["temp_x_winter"] = (df["temperature_F"].fillna(0) * df["season_winter"]).astype(float)
    df["snow_x_weekday"] = (df["snowfall_inch"].fillna(0) * (df["day_of_week"] < 5)).astype(float)
    for c in PROFILE_FEATURES:  # placeholders; real values come from add_station_profiles
        df[c] = 0.0
    return df.dropna(subset=STATION_FEATURES + config.TARGETS)


def add_station_profiles(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Historical profiles fit on the TRAINING years only, then mapped onto all three splits (in place):
    mean by hour-of-week and by month-hour, the station's overall mean, and last week's value relative
    to the hour-of-week profile."""
    keys = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        keys[name] = (d["hour"] + 24 * d["day_of_week"],
                      d["month"].astype(str) + "_" + d["hour"].astype(str))
    for target in config.TARGETS:
        k = target.replace("_trips", "")
        how_mean = train.groupby(keys["train"][0])[target].mean().to_dict()
        mh_mean = train.groupby(keys["train"][1])[target].mean().to_dict()
        station_mean = float(train[target].mean())
        for name, d in (("train", train), ("val", val), ("test", test)):
            how, mh = keys[name]
            d[f"hist_hour_of_week_{k}"] = how.map(how_mean).fillna(0)
            d[f"hist_month_hour_{k}"] = mh.map(mh_mean).fillna(0)
            d[f"station_mean_{k}"] = station_mean
            hist = d[f"hist_hour_of_week_{k}"]
            d[f"ratio_lag168_hist_{k}"] = np.where(hist > 1e-6, d[f"lag_168h_{k}"] / (hist + 1e-6), 0)
