"""Сборка ядра app.PredictionService на источниках из Postgres."""
from __future__ import annotations

from typing import Callable
from zoneinfo import ZoneInfo

import psycopg

from app import PredictionService
from inference.config import InferenceSettings
from inference.config import connect as pg_connect
from inference.reference import metadata_loader
from inference.weather import PgWeatherClient, resolve_district_id


def build_service(
    settings: InferenceSettings,
    connect: Callable[[], psycopg.Connection] | None = None,
) -> PredictionService:
    connect = connect or pg_connect

    with connect() as conn:
        district_id = resolve_district_id(
            conn,
            settings.weather_district_id,
        )

    weather = PgWeatherClient(
        connect=connect,
        district_id=district_id,
        # Та же зона, что ядро передаёт HTTP weather-service.
        tz=ZoneInfo(settings.service.weather_timezone),
        ttl_seconds=settings.weather_ttl_seconds,
        gap_ttl_seconds=settings.weather_gap_ttl_seconds,
    )

    return PredictionService(
        config=settings.service,
        metadata_loader=metadata_loader(connect),
        weather_client=weather,
    )
