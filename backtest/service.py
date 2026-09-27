"""Поднимает боевой app.PredictionService offline: те же parquet-справочники
и та же модель, что на стенде, но CSV-погода вместо HTTP weather-service и
без Postgres/pickle-state.
"""
from __future__ import annotations

from pathlib import Path

from app import PredictionService, ServiceConfig

from backtest.weather import CsvWeatherClient


def build_offline_service(
    repo_root: Path,
    weather_csv: Path,
    runtime_dir: Path,
) -> PredictionService:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    config = ServiceConfig(
        model_path=repo_root / "data" / "best_model.cbm",
        sensors_metadata_path=(
            repo_root / "data" / "справочник_каналов_датчиков.parquet"
        ),
        objects_metadata_path=(
            repo_root / "data" / "справочник_объектов_диспетчер.parquet"
        ),
        metadata_cache_path=runtime_dir / "metadata_cache.pkl",
        # Путь не существует: чужой/устаревший rolling-state не должен
        # подмешаться к бэктесту (см. app.load_state_if_exists).
        state_path=runtime_dir / "no_state.pkl",
        log_path=runtime_dir / "service.log",
    )

    return PredictionService(
        config=config,
        weather_client=CsvWeatherClient(weather_csv),
    )
