"""Offline WeatherSource для app.PredictionService: тот же CSV, что и в
retraining (open-meteo), вместо HTTP weather-service/Postgres.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import polars as pl

WEATHER_COLUMNS = (
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "precipitation_probability (%)",
    "precipitation (mm)",
    "rain (mm)",
    "snowfall (cm)",
    "snow_depth (m)",
    "weather_code (wmo code)",
    "cloud_cover (%)",
    "pressure_msl (hPa)",
)


class CsvWeatherClient:
    """Как app.WeatherSource: .get(hour) -> {feature_name: value}.

    Время CSV — UTC; сдвиг +3ч на МСК воспроизводит EXPECTED_WEATHER из
    retraining/notebooks/1_prepare_dataset.ipynb дословно.
    """

    def __init__(self, csv_path: Path) -> None:
        df = pl.read_csv(csv_path).with_columns(
            (pl.col("time").str.to_datetime() + pl.duration(hours=3)).alias("time")
        )

        self._by_hour: dict[datetime, Mapping[str, float | int | None]] = {
            row["time"]: row for row in df.iter_rows(named=True)
        }

    def get(self, target_hour: datetime) -> Mapping[str, float | int | None]:
        hour = target_hour.replace(minute=0, second=0, microsecond=0)
        row = self._by_hour.get(hour)

        if row is None:
            return {name: None for name in WEATHER_COLUMNS}

        return row

    def close(self) -> None:
        pass
