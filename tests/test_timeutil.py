from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app import parse_datetime
from inference.timeutil import (
    model_hour_to_utc,
    to_model_fields,
    to_model_time,
)


def test_utc_evening_is_next_moscow_day() -> None:
    # 21:30 UTC 31 декабря — уже 00:30 1 января по Москве.
    ts = datetime(2025, 12, 31, 21, 30, 5, tzinfo=timezone.utc)

    assert to_model_fields(ts) == ("2026-01-01", "00:30:05")


def test_utc_midnight_is_three_am_moscow() -> None:
    ts = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert to_model_fields(ts) == ("2026-08-01", "03:00:00")


def test_fields_roundtrip_through_core_parser() -> None:
    ts = datetime(2026, 3, 1, 20, 59, 59, 250000, tzinfo=timezone.utc)
    date_value, time_value = to_model_fields(ts)

    # Дробные секунды не теряются: порядок логов в окне сохраняется.
    assert parse_datetime(date_value, time_value) == datetime(
        2026, 3, 1, 23, 59, 59, 250000
    )


def test_session_timezone_does_not_matter() -> None:
    # psycopg отдаёт timestamptz в зоне сессии; результат от неё не зависит.
    utc = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    vladivostok = utc.astimezone(ZoneInfo("Asia/Vladivostok"))

    assert to_model_time(utc) == to_model_time(vladivostok)
    assert to_model_time(utc) == datetime(2026, 8, 1, 15, 0)


def test_naive_input_rejected() -> None:
    with pytest.raises(ValueError):
        to_model_time(datetime(2026, 8, 1, 12, 0))

    with pytest.raises(ValueError):
        model_hour_to_utc(datetime(2026, 8, 1, 12, tzinfo=timezone.utc))


def test_model_hour_to_utc_uses_tzdata() -> None:
    assert model_hour_to_utc(datetime(2026, 8, 1, 2)) == datetime(
        2026, 7, 31, 23, tzinfo=timezone.utc
    )

    # В 2013 году Москва жила по UTC+4 — смещение не зашито.
    assert model_hour_to_utc(datetime(2013, 8, 1, 12)) == datetime(
        2013, 8, 1, 8, tzinfo=timezone.utc
    )


def test_hour_roundtrip() -> None:
    hour = datetime(2026, 1, 1, 0)

    assert to_model_time(model_hour_to_utc(hour)) == hour
    assert to_model_time(
        model_hour_to_utc(hour) + timedelta(minutes=59)
    ) == datetime(2026, 1, 1, 0, 59)
