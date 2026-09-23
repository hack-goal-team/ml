"""Погода для модели из таблицы weather вместо HTTP weather-service.

Таблицу пишет поллер бэкенда раз в час (ADR-017): valid_for в UTC,
прошлые часы удаляются, текущий час лежит снимком current.
"""
from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row

from inference.timeutil import MODEL_TIMEZONE, model_hour_to_utc

# Колонка weather (001, 013) → имя погодной фичи модели.
WEATHER_COLUMNS = {
    "temperature_c": "temperature_2m (°C)",
    "humidity_pct": "relative_humidity_2m (%)",
    "precipitation_probability_pct": "precipitation_probability (%)",
    "precipitation": "precipitation (mm)",
    "rain_mm": "rain (mm)",
    "snowfall_cm": "snowfall (cm)",
    "snow_depth_m": "snow_depth (m)",
    "weather_code": "weather_code (wmo code)",
    "cloud_cover_pct": "cloud_cover (%)",
    "pressure_hpa": "pressure_msl (hPa)",
}

# Точка, на которой обучалась модель (README, раздел о погоде).
MODEL_LATITUDE = 55.7558
MODEL_LONGITUDE = 37.6173

SELECT_HOUR = f"""
    SELECT is_forecast, valid_for, {", ".join(WEATHER_COLUMNS)}
    FROM weather
    WHERE district_id = %s AND valid_for >= %s AND valid_for < %s
"""

SELECT_DISTRICTS = "SELECT district_id, latitude, longitude FROM districts"

log = logging.getLogger("inference.weather")

WeatherFeatures = dict[str, float | int | None]


def pick_row(
    rows: Iterable[Mapping[str, Any]],
    hour_utc: datetime,
) -> Mapping[str, Any] | None:
    """Строка для часа [hour_utc, hour_utc + 1 ч) или None."""
    rows = list(rows)

    # Hourly-прогноз ровно на час — то, на чём обучалась модель.
    for row in rows:
        if row["is_forecast"] and row["valid_for"] == hour_utc:
            return row

    # Текущий час поллер из hourly выкидывает и держит снимок current
    # с valid_for, выровненным на 15 минут. Берём ближайший к началу часа.
    current = [row for row in rows if not row["is_forecast"]]

    return min(current, key=lambda row: row["valid_for"], default=None)


def to_features(
    row: Mapping[str, Any] | None,
) -> WeatherFeatures:
    """Все 10 ключей всегда: пропуск — None, модель видит NaN (ADR-019)."""
    features: WeatherFeatures = {}

    for column, feature in WEATHER_COLUMNS.items():
        value = None if row is None else row[column]
        features[feature] = (
            float(value) if isinstance(value, Decimal) else value
        )

    return features


def nearest_district(
    rows: Iterable[Mapping[str, Any]],
    latitude: float = MODEL_LATITUDE,
    longitude: float = MODEL_LONGITUDE,
) -> int:
    # Равнопромежуточная проекция: в пределах города точнее не нужно.
    scale = math.cos(math.radians(latitude))

    def distance(row: Mapping[str, Any]) -> float:
        d_lat = float(row["latitude"]) - latitude
        d_lon = (float(row["longitude"]) - longitude) * scale
        return d_lat * d_lat + d_lon * d_lon

    best = min(rows, key=distance, default=None)

    if best is None:
        raise RuntimeError("Table districts is empty")

    return int(best["district_id"])


def resolve_district_id(
    conn: psycopg.Connection,
    configured: int | None,
) -> int:
    """WEATHER_DISTRICT_ID, а без него — район, ближайший к точке модели."""
    if configured is not None:
        return configured

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(SELECT_DISTRICTS)
            rows = cur.fetchall()
    except psycopg.errors.InsufficientPrivilege as exc:
        raise RuntimeError(
            "Role has no SELECT on districts: set WEATHER_DISTRICT_ID "
            "or grant SELECT ON districts"
        ) from exc

    return nearest_district(rows)


class PgWeatherClient:
    """Клиент погоды для PredictionService(weather_client=...).

    Кеш с TTL здесь, а не в app.WeatherCache: тот держит час вечно,
    а прогноз поллер переписывает каждый час.
    """

    def __init__(
        self,
        connect: Callable[[], psycopg.Connection],
        district_id: int,
        tz: ZoneInfo = MODEL_TIMEZONE,
        ttl_seconds: float = 300.0,
        gap_ttl_seconds: float = 60.0,
        max_entries: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.connect = connect
        self.district_id = district_id
        self.tz = tz
        self.ttl_seconds = ttl_seconds
        self.gap_ttl_seconds = gap_ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._conn: psycopg.Connection | None = None
        self._cache: OrderedDict[
            datetime,
            tuple[float, WeatherFeatures],
        ] = OrderedDict()

    def get(
        self,
        target_hour: datetime,
    ) -> WeatherFeatures:
        hour = target_hour.replace(minute=0, second=0, microsecond=0)
        now = self.clock()

        cached = self._cache.get(hour)
        if cached is not None and cached[0] > now:
            self._cache.move_to_end(hour)
            return cached[1]

        features = self._fetch(hour)

        # Пропуск кешируем коротко: следующий заход поллера может его
        # закрыть, а без кеша догон по старым часам бил бы в БД на каждый лог.
        has_value = any(value is not None for value in features.values())
        ttl = self.ttl_seconds if has_value else self.gap_ttl_seconds

        self._cache[hour] = (now + ttl, features)
        self._cache.move_to_end(hour)

        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        return features

    def _fetch(
        self,
        hour: datetime,
    ) -> WeatherFeatures:
        hour_utc = model_hour_to_utc(hour, self.tz)

        try:
            if self._conn is None or self._conn.closed:
                self._conn = self.connect()
                # Без транзакции: не висим idle in transaction между часами.
                self._conn.autocommit = True

            with self._conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    SELECT_HOUR,
                    (
                        self.district_id,
                        hour_utc,
                        hour_utc + timedelta(hours=1),
                    ),
                )
                rows = cur.fetchall()
        except psycopg.Error as exc:
            # Сбой БД — тот же пропуск (ADR-019), а не 503 на весь прогноз.
            log.warning("weather query failed hour=%s: %s", hour, exc)
            self._reset()
            return to_features(None)

        return to_features(pick_row(rows, hour_utc))

    def _reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except psycopg.Error:
                pass
        self._conn = None

    def close(self) -> None:
        self._reset()
