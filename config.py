"""Paths and experiment constants shared by every script.

Set the environment variable BIKESHARE_DATA_DIR to use a data folder other than ./data.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BIKESHARE_DATA_DIR", ROOT / "data"))

# --- Raw inputs (see data/README.md for where to download them) ---
RAW_TRIPS_DIR = DATA_DIR / "raw" / "trips"             # one sub-folder per year with monthly trip CSVs
WEATHER_CSV = DATA_DIR / "raw" / "weather_open_meteo.csv"
HOLIDAYS_CSV = DATA_DIR / "raw" / "us_federal_holidays_2018_2025.csv"

# --- Processed data written by scripts/01_prepare_data.py ---
PROCESSED_DIR = DATA_DIR / "processed"
SYSTEM_HOURLY_CSV = PROCESSED_DIR / "system_hourly_trips.csv"            # trips only
SYSTEM_WEATHER_CSV = PROCESSED_DIR / "system_hourly_weather.csv"         # trips + weather (model input)
STATION_HOURLY_CSV = PROCESSED_DIR / "station_hourly_trips.csv"          # all stations, large intermediate
STATION_DIR = PROCESSED_DIR / "stations"                                 # one CSV per top-N station
TOP_STATIONS_CSV = STATION_DIR / "top50_stations.csv"

# --- Outputs ---
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --- Experiment design ---
TARGETS = ["outgoing_trips", "incoming_trips"]
TRAIN_YEARS = (2018, 2023)          # inclusive
VAL_YEAR = 2024
TEST_YEAR = 2025
HORIZONS = [1, 4, 8, 12, 24]        # hours ahead
RANDOM_STATE = 42

YEAR_MIN, YEAR_MAX = 2018, 2025     # trip records outside this range are data-entry errors
TOP_N_STATIONS = 50
# A station must have real trips up to this date to be modeled; otherwise zero-filling its missing
# hours would fabricate easy-to-predict "zero demand" for a closed station.
DISCONTINUED_CUTOFF = f"{TEST_YEAR}-11-15"
COVID_START, COVID_END = "2020-03-01", "2021-12-31"

# --- Zero-shot foundation model ---
TIMESFM_MODEL_ID = "google/timesfm-2.0-500m-pytorch"
TIMESFM_HIGH_FREQ = 0               # TimesFM frequency code for hourly (high-frequency) data
TIMESFM_MEDIUM_FREQ = 1             # used only to reproduce the flawed legacy protocol
