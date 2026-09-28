"""Four-class incident routing and previous target history."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping

import psycopg

SENSOR_CLASSES = json.loads(
    (Path(__file__).resolve().parent.parent / "data/incident4/incident_by_sensor.json")
    .read_text(encoding="utf-8")
)
INCIDENT_VALUES = {
    "FIRE": frozenset((
        "Обнаружен дым", "Рычаг сдернут", "Рычаг сдернут влево",
        "Рычаг сдернут вправо", "Оба рычага сдернуты",
    )),
    "FLOOD": frozenset((
        "Затоплен", "Работают все насосы в АНС", "Работают все насосы АНС",
        "Включены все насосы АНС",
    )),
    "GAS": frozenset(("Обнаружен газ",)),
    "INTRUSION": frozenset((
        "Обнаружено движение", "Движение вверх", "Движение вниз",
        "Движение влево", "Движение вправо",
    )),
}
OPEN_CONTACT = frozenset((
    "Ручной извещатель", "Тепловой датчик", "Датчик дыма",
    "Состояние УИР-Р", "Датчик затопления", "КД Дверь", "КД Люк",
    "КД АВ", "Стекло", "9-секционный люк",
))

SELECT_LAST_TARGETS = """
    SELECT DISTINCT ON (e.channel_id) e.channel_id, e.ts, e.id
    FROM events e
    JOIN dim_channels_current c ON c.channel_id = e.channel_id
    WHERE e.is_alarm
      AND (e.ts < timestamptz '2021-01-01 00:00:00+03'
           OR e.ts >= timestamptz '2022-01-01 00:00:00+03')
      AND (e.id <= %(low)s OR e.id = ANY(%(seen)s::bigint[]))
      AND (
        (c.sensor_type = ANY(%(fire)s::text[]) AND e.raw_value = ANY(%(fire_values)s::text[]))
        OR (c.sensor_type = ANY(%(flood)s::text[]) AND e.raw_value = ANY(%(flood_values)s::text[]))
        OR (c.sensor_type = ANY(%(gas)s::text[]) AND e.raw_value = ANY(%(gas_values)s::text[]))
        OR (c.sensor_type = ANY(%(intrusion)s::text[]) AND e.raw_value = ANY(%(intrusion_values)s::text[]))
        OR (c.sensor_type = ANY(%(open_contact)s::text[]) AND e.raw_value = 'Не замкнут')
      )
    ORDER BY e.channel_id, e.ts DESC, e.id DESC
"""


def incident_class(sensor_type: str | None) -> str | None:
    return SENSOR_CLASSES.get(sensor_type)


def is_target_alarm(sensor_type: str | None, value: str, is_alarm: bool) -> bool:
    kind = incident_class(sensor_type)
    return bool(
        is_alarm and kind and (
            value in INCIDENT_VALUES[kind]
            or (value == "Не замкнут" and sensor_type in OPEN_CONTACT)
        )
    )


@dataclass
class TargetHistory:
    latest: dict[int, tuple[datetime, int]] = field(default_factory=dict)

    def hours_before(self, channel_id: int, at: datetime) -> float | None:
        previous = self.latest.get(channel_id)
        if previous is None or previous[0] > at:
            return None
        return (at - previous[0]).total_seconds() / 3600.0

    def mark(self, channel_id: int, at: datetime, event_id: int) -> None:
        previous = self.latest.get(channel_id)
        if previous is None or (at, event_id) > previous:
            self.latest[channel_id] = (at, event_id)

    def save(self, path: Path, low: int, seen: set[int]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        data = {
            "low": low,
            "seen": sorted(seen),
            "latest": {str(k): [ts.isoformat(), event_id]
                       for k, (ts, event_id) in self.latest.items()},
        }
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, low: int, seen: set[int]) -> "TargetHistory | None":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["low"] != low or set(data["seen"]) != seen:
                return None
            return cls({int(k): (datetime.fromisoformat(v[0]), int(v[1]))
                        for k, v in data["latest"].items()})
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return None

    @classmethod
    def from_db(cls, conn: psycopg.Connection, low: int,
                seen: set[int]) -> "TargetHistory":
        marker = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM alarm_backfill)"
        ).fetchone()[0]
        if not marker:
            raise RuntimeError("alarm_backfill must finish before incident inference")
        params: dict[str, object] = {"low": low, "seen": list(seen),
                                    "open_contact": list(OPEN_CONTACT)}
        for kind in INCIDENT_VALUES:
            params[kind.lower()] = [sensor for sensor, value in SENSOR_CLASSES.items()
                                    if value == kind]
            params[kind.lower() + "_values"] = list(INCIDENT_VALUES[kind])
        with conn.transaction():
            conn.execute("SET LOCAL statement_timeout = '30min'")
            with conn.cursor(name="incident_targets") as cur:
                cur.itersize = 10_000
                cur.execute(SELECT_LAST_TARGETS, params)
                return cls({int(channel): (at, int(event_id))
                            for channel, at, event_id in cur})
