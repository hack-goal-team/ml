"""Настройки inference из переменных окружения."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import psycopg

from app import ServiceConfig, normalize_horizon_hours

# Подключение — стандартные переменные libpq, psycopg читает их сам.
# Пароль может прийти из PGPASSWORD, .pgpass/PGPASSFILE или PGSERVICE.
REQUIRED_PG_ENV = ("PGHOST", "PGDATABASE", "PGUSER")

# Без таймаутов оборванный сокет вешает execute навсегда (как в моке).
CONNECT_KWARGS = dict(
    connect_timeout=10,
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=3,
)
STATEMENT_TIMEOUT = "-c statement_timeout=30s"


def model_version(model_path: Path) -> str:
    """catboost-aft-<первые 12 hex sha256 файла модели>."""
    digest = hashlib.sha256()

    with model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return f"catboost-aft-{digest.hexdigest()[:12]}"


@dataclass(slots=True, frozen=True)
class InferenceSettings:
    # Конфиг ядра: модель, runtime-пути, SHAP-порог, логирование.
    service: ServiceConfig

    # Горизонт прогноза: 24 ч плюс запас на молчащий канал.
    horizon_hours: float

    # Район погоды; None — ближайший к точке обучения модели.
    weather_district_id: int | None

    # Сколько секунд живёт найденный час погоды и час-пропуск.
    weather_ttl_seconds: float
    weather_gap_ttl_seconds: float

    # Пишется в prediction_log.model_version.
    model_version: str

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] = os.environ,
    ) -> "InferenceSettings":
        missing = [name for name in REQUIRED_PG_ENV if not env.get(name)]
        if missing and not env.get("PGSERVICE"):
            raise RuntimeError(
                f"Missing environment variables: {', '.join(missing)}"
            )

        service = ServiceConfig.from_yaml(env.get("ML_CONFIG", "config.yml"))

        if env.get("MODEL_PATH"):
            service = replace(service, model_path=Path(env["MODEL_PATH"]))

        # Кеш ядра хранит час вечно; с 0 он пропускает всё насквозь,
        # а свежесть погоды держит TTL-кеш PgWeatherClient.
        service = replace(service, weather_cache_max_entries=0)

        district = env.get("WEATHER_DISTRICT_ID")

        return cls(
            service=service,
            horizon_hours=normalize_horizon_hours(
                env.get("HORIZON_HOURS", "30")
            ),
            weather_district_id=int(district) if district else None,
            weather_ttl_seconds=float(
                env.get("WEATHER_TTL_SECONDS", "300")
            ),
            weather_gap_ttl_seconds=float(
                env.get("WEATHER_GAP_TTL_SECONDS", "60")
            ),
            model_version=(
                env.get("MODEL_VERSION")
                or model_version(service.model_path)
            ),
        )


def pg_options(env: Mapping[str, str] = os.environ) -> str:
    # Аргумент options перекрывает PGOPTIONS целиком, поэтому склеиваем;
    # PGOPTIONS идёт последним, и его statement_timeout побеждает.
    return f"{STATEMENT_TIMEOUT} {env.get('PGOPTIONS', '')}".strip()


def connect() -> psycopg.Connection:
    """Подключение под ролью из PGUSER (inference, ADR-025)."""
    return psycopg.connect(
        autocommit=True,
        options=pg_options(),
        **CONNECT_KWARGS,
    )
