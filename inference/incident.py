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
SEED_PATH = Path(__file__).resolve().parent.parent / "data/incident4/target_seed.json"
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

SELECT_SINCE_CHECKPOINT = """
    SELECT e.channel_id, e.ts, e.id, e.raw_value, e.is_alarm,
           e.journal_is_alarm, c.sensor_type
    FROM events e
    JOIN dim_channels_current c ON c.channel_id = e.channel_id
    WHERE e.ts >= %(cutoff)s
      AND (e.id <= %(low)s OR e.id = ANY(%(seen)s::bigint[]))
      AND c.sensor_type = ANY(%(sensors)s::text[])
      AND (
        e.raw_value = ANY(%(values)s::text[])
        OR e.raw_value = 'Не замкнут'
      )
    ORDER BY e.channel_id, e.ts, e.id
"""


def classed_channels(conn: psycopg.Connection) -> frozenset[int]:
    """Каналы с классом инцидента — по реестру, а не по фичам модели."""
    rows = conn.execute(
        "SELECT channel_id FROM dim_channels_current WHERE sensor_type = ANY(%s)",
        (list(SENSOR_CLASSES),),
    ).fetchall()
    return frozenset(int(row[0]) for row in rows)


def incident_class(sensor_type: str | None) -> str | None:
    return SENSOR_CLASSES.get(sensor_type)


def is_target_alarm(sensor_type: str | None, value: str, is_alarm: bool,
                    journal_alarm: bool | None = None) -> bool:
    kind = incident_class(sensor_type)
    candidate = bool(kind and (
        value in INCIDENT_VALUES[kind]
        or (value == "Не замкнут" and sensor_type in OPEN_CONTACT)
    ))
    if not candidate:
        return False
    alarm = journal_alarm if kind == "INTRUSION" else is_alarm
    if kind == "INTRUSION" and journal_alarm is None:
        raise RuntimeError("Original intrusion alarm flag is missing")
    return bool(alarm)


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
                seen: set[int], seed_path: Path = SEED_PATH) -> "TargetHistory":
        marker = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM alarm_backfill)"
        ).fetchone()[0]
        if not marker:
            raise RuntimeError("alarm_backfill must finish before incident inference")
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        latest = cls({int(channel): (datetime.fromisoformat(row[0]), int(row[1]))
                      for channel, row in seed["latest"].items()})
        checkpoint = conn.execute(
            "SELECT covered_until, targets FROM incident_history_checkpoint WHERE id = 1"
        ).fetchone()
        if checkpoint is None:
            raise RuntimeError("incident history checkpoint is missing")
        cutoff, targets = checkpoint
        if cutoff < datetime.fromisoformat(seed["cutoff"]):
            raise RuntimeError("incident history checkpoint predates model seed")
        for channel, row in targets.items():
            latest.mark(int(channel), datetime.fromisoformat(row[0]), int(row[1]))
        params = {
            "cutoff": cutoff, "low": low, "seen": list(seen),
            "sensors": list(SENSOR_CLASSES),
            "values": list(set().union(*INCIDENT_VALUES.values())),
        }
        with conn.transaction():
            conn.execute("SET LOCAL statement_timeout = '10min'")
            with conn.cursor(name="incident_targets") as cur:
                cur.itersize = 10_000
                cur.execute(SELECT_SINCE_CHECKPOINT, params)
                for channel, at, event_id, value, alarm, journal, sensor in cur:
                    if is_target_alarm(sensor, value, alarm, journal):
                        latest.mark(int(channel), at, int(event_id))
        return latest
