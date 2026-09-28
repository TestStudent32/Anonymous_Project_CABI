"""Step 1: raw trip files + weather -> processed hourly inputs.

Writes (under data/processed/):
  system_hourly_trips.csv     system-wide hourly counts
  system_hourly_weather.csv   + weather (input to all system-level models and the station grid)
  station_hourly_trips.csv    every station, hourly (large, ~1.2 GB)
  stations/top50_stations.csv and stations/station_<id>.csv for the 50 busiest stations

Usage:  python scripts/01_prepare_data.py [--skip-stations]
Runtime: roughly 30-60 min on a laptop (dominated by reading ~5 GB of trip CSVs twice).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import data_prep  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stations", action="store_true", help="Only build the system-wide files")
    args = parser.parse_args()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather = data_prep.load_weather()

    print("[1/3] System-wide hourly trips")
    trips = data_prep.extract_system_hourly()
    trips.to_csv(config.SYSTEM_HOURLY_CSV, index=False)
    data_prep.build_system_file(trips, weather).to_csv(config.SYSTEM_WEATHER_CSV, index=False)
    print(f"  -> {config.SYSTEM_WEATHER_CSV}")
    if args.skip_stations:
        return

    print("[2/3] Per-station hourly trips")
    data_prep.extract_station_hourly().to_csv(config.STATION_HOURLY_CSV, index=False)

    print(f"[3/3] Top {config.TOP_N_STATIONS} stations -> one file each")
    top = data_prep.top_stations()
    config.STATION_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(config.TOP_STATIONS_CSV, index=False)
    data_prep.build_station_files(top, weather)


if __name__ == "__main__":
    main()
