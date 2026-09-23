"""Перевод времени между базой (timestamptz) и моделью (наивное МСК)."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Модель обучена на наивном времени Europe/Moscow: дата/время
# журнала и часы погоды Open-Meteo с timezone=Europe/Moscow.
MODEL_TIMEZONE = ZoneInfo("Europe/Moscow")


def to_model_time(
    ts: datetime,
    tz: ZoneInfo = MODEL_TIMEZONE,
) -> datetime:
    # Наивное время без зоны неоднозначно: events.ts всегда timestamptz.
    if ts.tzinfo is None:
        raise ValueError(f"Expected aware datetime, got {ts!r}")

    return ts.astimezone(tz).replace(tzinfo=None)


def to_model_fields(
    ts: datetime,
    tz: ZoneInfo = MODEL_TIMEZONE,
) -> tuple[str, str]:
    """events.ts → значения полей `дата` и `время` для predict()."""
    local = to_model_time(ts, tz)

    return (
        local.date().isoformat(),
        local.time().isoformat(),
    )


def model_hour_to_utc(
    hour: datetime,
    tz: ZoneInfo = MODEL_TIMEZONE,
) -> datetime:
    """Наивный час модели → момент UTC, как его пишет поллер погоды."""
    if hour.tzinfo is not None:
        raise ValueError(f"Expected naive model hour, got {hour!r}")

    # Смещение берётся из tzdata, а не зашито +3: правила зоны
    # менялись (UTC+4 в 2011-2014) и могут поменяться снова.
    return hour.replace(tzinfo=tz).astimezone(timezone.utc)
