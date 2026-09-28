from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from catboost import CatBoostRegressor

from app import WeatherCache, compile_model_features
from inference.weather import (
    WEATHER_COLUMNS,
    PgWeatherClient,
    nearest_district,
    pick_row,
    resolve_district_id,
    to_features,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UTC = timezone.utc
HOUR = datetime(2026, 9, 24, 11, tzinfo=UTC)  # 14:00 МСК


def _row(valid_for, is_forecast, temperature=Decimal("12.40")):
    row = dict.fromkeys(WEATHER_COLUMNS)
    row.update(
        is_forecast=is_forecast,
        valid_for=valid_for,
        temperature_c=temperature,
    )
    return row


def test_keys_are_exactly_model_weather_features() -> None:
    model = CatBoostRegressor()
    model.load_model(str(DATA / "best_model.cbm"))
    levels = json.loads((DATA / "feature_encoding.json").read_text(encoding="utf-8"))["levels"]
    compiled = compile_model_features(model.feature_names_, levels)

    expected = {spec.name for spec in compiled.specs if spec.kind == "weather"}

    assert set(to_features(None)) == expected
    assert len(expected) == 10


def test_forecast_for_exact_hour_wins() -> None:
    rows = [
        _row(HOUR + timedelta(minutes=15), False, Decimal("1")),
        _row(HOUR, True, Decimal("2")),
    ]

    assert pick_row(rows, HOUR)["temperature_c"] == Decimal("2")


def test_current_snapshot_covers_current_hour() -> None:
    rows = [
        _row(HOUR + timedelta(minutes=45), False, Decimal("3")),
        _row(HOUR + timedelta(minutes=15), False, Decimal("1")),
    ]

    assert pick_row(rows, HOUR)["temperature_c"] == Decimal("1")


def test_no_rows_is_gap() -> None:
    assert pick_row([], HOUR) is None
    assert to_features(None) == dict.fromkeys(WEATHER_COLUMNS.values())


def test_values_converted_for_model() -> None:
    row = _row(HOUR, True)
    row.update(weather_code=61, snow_depth_m=None)
    features = to_features(row)

    assert features["temperature_2m (°C)"] == 12.4
    assert type(features["temperature_2m (°C)"]) is float
    assert features["weather_code (wmo code)"] == 61
    assert features["snow_depth (m)"] is None


def test_nearest_district() -> None:
    rows = [
        {"district_id": i, "latitude": Decimal(lat), "longitude": Decimal(lon)}
        for i, lat, lon in [
            (1, "55.9", "37.5"),
            (2, "55.75583", "37.61778"),
            (3, "55.6", "37.7"),
        ]
    ]

    assert nearest_district(rows) == 2

    with pytest.raises(RuntimeError):
        nearest_district([])


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _client(results, clock, **kwargs) -> tuple[PgWeatherClient, list]:
    calls = []
    client = PgWeatherClient(
        connect=lambda: None,
        district_id=1,
        clock=clock,
        **kwargs,
    )

    def fetch(hour):
        calls.append(hour)
        return results.pop(0)

    client._fetch = fetch
    return client, calls


def test_cache_expires_after_ttl() -> None:
    clock = _Clock()
    old = {"temperature_2m (°C)": 1.0}
    fresh = {"temperature_2m (°C)": 2.0}
    client, calls = _client([old, fresh], clock, ttl_seconds=300)
    hour = datetime(2026, 9, 24, 14, 37)

    assert client.get(hour) is old
    clock.now += 299
    assert client.get(hour) is old
    clock.now += 2
    assert client.get(hour) is fresh
    assert calls == [datetime(2026, 9, 24, 14)] * 2


def test_gap_cached_shortly() -> None:
    clock = _Clock()
    gap = to_features(None)
    fresh = {"temperature_2m (°C)": 2.0}
    client, calls = _client(
        [gap, fresh],
        clock,
        ttl_seconds=300,
        gap_ttl_seconds=60,
    )
    hour = datetime(2026, 9, 24, 14)

    assert client.get(hour) is gap
    clock.now += 59
    assert client.get(hour) is gap
    clock.now += 2
    assert client.get(hour) is fresh
    assert len(calls) == 2


def test_cache_bounded() -> None:
    clock = _Clock()
    client, calls = _client(
        [{"x": float(i)} for i in range(4)],
        clock,
        max_entries=2,
    )
    base = datetime(2026, 9, 24, 0)

    for i in range(3):
        client.get(base + timedelta(hours=i))

    assert len(client._cache) == 2
    client.get(base)
    assert len(calls) == 4


def test_core_cache_with_zero_entries_passes_through() -> None:
    # Режим ядра для inference: WeatherCache не пинит час навсегда.
    calls = []

    class Source:
        def get(self, hour):
            calls.append(hour)
            return {}

    cache = WeatherCache(client=Source(), max_entries=0)
    cache.get(datetime(2026, 9, 24, 14, 5))
    cache.get(datetime(2026, 9, 24, 14, 50))

    assert len(calls) == 2


def test_db_failure_is_gap_and_backs_off() -> None:
    clock = _Clock()
    attempts = []

    def connect():
        attempts.append(clock.now)
        raise psycopg.OperationalError("connection refused")

    client = PgWeatherClient(
        connect=connect,
        district_id=1,
        gap_ttl_seconds=60,
        clock=clock,
    )
    base = datetime(2026, 9, 24, 14)

    # Все 8 смещений прогноза — одна попытка, а не 8 по connect_timeout.
    for offset in (0, 1, 4, 8, 12, 16, 20, 24):
        hour = base + timedelta(hours=offset)
        assert client.get(hour) == to_features(None)
    assert len(attempts) == 1

    clock.now += 59
    client.get(base + timedelta(hours=2))
    assert len(attempts) == 1

    clock.now += 2
    client.get(base + timedelta(hours=3))
    assert len(attempts) == 2
    client.close()


def _insert_weather(conn, rows) -> None:
    columns = ["district_id", "fetched_at", *rows[0]]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO weather ({', '.join(columns)}) "
            f"VALUES ({', '.join(['%s'] * len(columns))})",
            [(5773, HOUR, *row.values()) for row in rows],
        )


def test_reads_postgres_as_poller_writes(pg) -> None:
    with pg.admin() as conn:
        conn.execute(
            "INSERT INTO districts VALUES "
            "(5773, 'Район', 55.75583, 37.61778), (1, 'Далеко', 59.9, 30.3)"
        )
        # Заход поллера в 14:20 МСК: current на 14:15 и hourly с 14:00
        # (backend#47 хранит hourly и для текущего часа).
        _insert_weather(
            conn,
            [
                {"valid_for": HOUR + timedelta(minutes=15),
                 "is_forecast": False, "temperature_c": Decimal("10.5"),
                 "weather_code": 3},
                {"valid_for": HOUR,
                 "is_forecast": True, "temperature_c": Decimal("10.00"),
                 "weather_code": 2},
                {"valid_for": HOUR + timedelta(hours=1),
                 "is_forecast": True, "temperature_c": Decimal("11.25"),
                 "weather_code": 61},
            ],
        )

    with pg.inference() as conn:
        assert resolve_district_id(conn, 7) == 7
        assert resolve_district_id(conn, None) == 5773

    client = PgWeatherClient(connect=pg.inference, district_id=5773)

    # Есть и current, и hourly на текущий час — побеждает hourly.
    current = client.get(datetime(2026, 9, 24, 14, 37))
    assert current["temperature_2m (°C)"] == 10.0
    assert current["weather_code (wmo code)"] == 2

    ahead = client.get(datetime(2026, 9, 24, 15))
    assert ahead["temperature_2m (°C)"] == 11.25
    assert ahead["weather_code (wmo code)"] == 61
    assert ahead["cloud_cover (%)"] is None

    # Прошлый час поллер удалил, будущий за горизонтом не пришёл.
    assert client.get(datetime(2026, 9, 24, 13)) == to_features(None)
    assert client.get(datetime(2026, 9, 26, 15)) == to_features(None)
    client.close()

    with pg.admin() as conn:
        conn.execute("DELETE FROM weather WHERE is_forecast AND valid_for = %s",
                     (HOUR,))
        conn.execute("REVOKE SELECT ON districts FROM inference")

    # Hourly на текущий час нет — запасной вариант, снимок current.
    fallback = PgWeatherClient(connect=pg.inference, district_id=5773)
    assert fallback.get(datetime(2026, 9, 24, 14))[
        "temperature_2m (°C)"
    ] == 10.5
    fallback.close()

    with pg.inference() as conn:
        with pytest.raises(RuntimeError, match="WEATHER_DISTRICT_ID"):
            resolve_district_id(conn, None)
