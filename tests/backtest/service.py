"""Боевой app.PredictionService offline: модель и справочники стенда,
CSV-погода Open-Meteo вместо weather-service, без Postgres и pickle-state.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from app import PredictionService, ServiceConfig


class CsvWeatherClient:
    """app.WeatherSource из CSV; UTC +3ч, как EXPECTED_WEATHER в ноутбуке 1."""

    def __init__(self, csv_path: Path) -> None:
        df = pl.read_csv(csv_path).with_columns(
            (pl.col("time").str.to_datetime() + pl.duration(hours=3)).alias("time")
        )
        self._by_hour = {row["time"]: row for row in df.iter_rows(named=True)}
        self._missing = dict.fromkeys(df.columns)

    def get(self, target_hour: datetime) -> dict:
        hour = target_hour.replace(minute=0, second=0, microsecond=0)
        return self._by_hour.get(hour, self._missing)

    def close(self) -> None:
        pass


def build_offline_service(repo_root: Path, weather_csv: Path) -> PredictionService:
    # Свежий каталог: чужой metadata-кеш и rolling-state не подмешаются.
    runtime_dir = Path(tempfile.mkdtemp(prefix="hack174-backtest-"))
    config = ServiceConfig(
        model_path=repo_root / "data" / "best_model.cbm",
        sensors_metadata_path=repo_root / "data" / "справочник_каналов_датчиков.parquet",
        objects_metadata_path=repo_root / "data" / "справочник_объектов_диспетчер.parquet",
        metadata_cache_path=runtime_dir / "metadata_cache.pkl",
        state_path=runtime_dir / "no_state.pkl",
        log_path=runtime_dir / "service.log",
    )
    return PredictionService(config=config, weather_client=CsvWeatherClient(weather_csv))
