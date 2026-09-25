from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from app import DATE_COL, ID_COL, TIME_COL, VALUE_COL
from inference.config import InferenceSettings, pg_options
from inference.service import build_service
from inference.timeutil import to_model_fields
from inference.weather import PgWeatherClient

ROOT = Path(__file__).resolve().parent.parent
PG_ENV = {
    "PGHOST": "db",
    "PGDATABASE": "goal",
    "PGUSER": "inference",
    "PGPASSWORD": "secret",
    "ML_CONFIG": str(ROOT / "config.yml"),
    "MODEL_PATH": str(ROOT / "data" / "best_model.cbm"),
}


def test_settings_defaults() -> None:
    settings = InferenceSettings.from_env(PG_ENV)
    digest = hashlib.sha256(
        (ROOT / "data" / "best_model.cbm").read_bytes()
    ).hexdigest()

    assert settings.horizon_hours == 30.0
    assert settings.weather_district_id is None
    assert settings.model_version == f"catboost-aft-{digest[:12]}"
    assert settings.service.weather_cache_max_entries == 0


def test_settings_overrides() -> None:
    settings = InferenceSettings.from_env(
        {
            **PG_ENV,
            "HORIZON_HOURS": "24",
            "WEATHER_DISTRICT_ID": "5773",
            "MODEL_VERSION": "catboost-aft-test",
        }
    )

    assert settings.horizon_hours == 24.0
    assert settings.weather_district_id == 5773
    assert settings.model_version == "catboost-aft-test"


def test_settings_require_pg_env() -> None:
    # Пароль может прийти из .pgpass/PGPASSFILE — он необязателен.
    no_password = {k: v for k, v in PG_ENV.items() if k != "PGPASSWORD"}
    InferenceSettings.from_env(no_password)

    no_host = {k: v for k, v in no_password.items() if k != "PGHOST"}
    with pytest.raises(RuntimeError, match="PGHOST"):
        InferenceSettings.from_env(no_host)
    InferenceSettings.from_env({**no_host, "PGSERVICE": "goal"})

    with pytest.raises(ValueError):
        InferenceSettings.from_env({**PG_ENV, "HORIZON_HOURS": "0"})


def test_pg_options_keep_pgoptions() -> None:
    assert pg_options({}) == "-c statement_timeout=30s"
    assert pg_options({"PGOPTIONS": "-c statement_timeout=5s"}) == (
        "-c statement_timeout=30s -c statement_timeout=5s"
    )


def test_prediction_on_postgres_sources(pg, tmp_path: Path) -> None:
    settings = InferenceSettings.from_env(
        {**PG_ENV, "WEATHER_DISTRICT_ID": "5773"}
    )
    settings = replace(
        settings,
        service=replace(
            settings.service,
            state_path=tmp_path / "state.pkl",
            log_path=tmp_path / "service.log",
        ),
    )
    snapshot = datetime(2026, 9, 1, tzinfo=timezone.utc)
    event_ts = datetime(2026, 9, 24, 11, 37, 15, tzinfo=timezone.utc)

    with pg.admin() as conn:
        conn.execute(
            "INSERT INTO districts VALUES (5773, 'Район', 55.75583, 37.61778)"
        )
        conn.execute(
            "INSERT INTO dim_objects VALUES (20, %s, 3, 5773, 'controlHouse', 'x')",
            (snapshot,),
        )
        conn.execute(
            "INSERT INTO dim_channels VALUES "
            "(120578, %s, 'Охранная', 'КД АВ', '15-11.1.131.2.', 'КД АВ', 20)",
            (snapshot,),
        )
        conn.execute(
            "INSERT INTO weather (district_id, valid_for, fetched_at, "
            "is_forecast, temperature_c) VALUES (5773, %s, %s, false, 9.75)",
            (datetime(2026, 9, 24, 11, 15, tzinfo=timezone.utc), event_ts),
        )

    service = build_service(settings, connect=pg.inference)

    try:
        assert isinstance(service.weather_client, PgWeatherClient)
        assert service.metadata == {
            120578: {
                "тип_датчика": "КД АВ",
                "тег_инженерной_системы": "15",
                "родитель": 5773,
            }
        }

        date_value, time_value = to_model_fields(event_ts)
        result = service.predict(
            {
                ID_COL: 120578,
                DATE_COL: date_value,
                TIME_COL: time_value,
                VALUE_COL: "Норма",
                "horizon_hours": settings.horizon_hours,
            }
        )
        assert result["datetime"] == "2026-09-24T14:37:15"
        assert 0.0 <= result["probability"] <= 1.0

        # В строку фичей попала погода из PG, недостающее — NaN.
        row = service._build_feature_row(
            120578,
            datetime(2026, 9, 24, 14, 37, 15),
        )
        names = service.feature_names
        current = row[names.index("temperature_2m (°C)__future_current")]
        ahead = row[names.index("temperature_2m (°C)__future_1h")]
        assert current == float(Decimal("9.75"))
        assert math.isnan(ahead)
    finally:
        service.close()
