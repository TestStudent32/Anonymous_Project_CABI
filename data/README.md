# Data (not included; all public)

Place the raw files as follows, then run `python scripts/01_prepare_data.py`:

```
data/raw/trips/<year>/*.csv                 Capital Bikeshare monthly trip histories, 2018-2025
data/raw/weather_open_meteo.csv             Open-Meteo hourly historical weather export
data/raw/us_federal_holidays_2018_2025.csv  columns: Date, Day, Name, Type, Details
```

- **Trips:** https://capitalbikeshare.com/system-data (monthly zip files; unzip each year into its own folder).
- **Weather:** https://open-meteo.com/en/docs/historical-weather-api — location 38.91N, 77.07W,
  2018-01-01 to 2025-12-31, hourly, local time (America/New_York), units °F / mph / inch. Variables:
  temperature_2m, snowfall, wind_speed_10m, relative_humidity_2m, apparent_temperature, precipitation,
  rain, cloud_cover, wind_speed_100m, is_day, direct_radiation. Save the CSV export as-is (3 metadata lines + header).
- **Holidays:** U.S. federal holidays 2018-2025 (104 dates), one row per holiday.
- **Forecast weather** (`data/raw/weather_forecast_previous_day1.csv`): Open-Meteo Previous Runs API,
  https://previous-runs-api.open-meteo.com/v1/forecast with latitude=38.910366, longitude=-77.07251,
  start_date=2024-01-01, end_date=2025-12-31, timezone=America/New_York, °F / mph / inch, format=csv, and
  hourly = temperature_2m, apparent_temperature, relative_humidity_2m, wind_speed_10m, precipitation, rain,
  snowfall, cloud_cover, direct_radiation, each with the suffix `_previous_day1`.

`data/processed/` is created by step 1.
