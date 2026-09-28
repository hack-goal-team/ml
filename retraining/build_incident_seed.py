"""Build previous-target snapshot from the original 2019–2026 journal."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ML/homemade/hack128/output_channel_dedup"
SENSORS = ROOT / "ML/retraining/data/reference/справочник_каналов_датчиков.parquet"
STATES = ROOT / "Sources/dataset/справочник_состояний.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data/incident4/target_seed.json"
CUTOFF = datetime(2026, 7, 1, tzinfo=ZoneInfo("Europe/Moscow"))


def main() -> None:
    mapping = json.loads(OUTPUT.with_name("incident_by_sensor.json").read_text())
    states = pl.read_csv(STATES).group_by("название_состояния").agg(
        pl.col("тревожное").unique()
    )
    alarm_by_value = {name: flags[0] for name, flags in states.iter_rows()
                      if len(flags) == 1}
    incident_values = {
        "FIRE": ["Обнаружен дым", "Рычаг сдернут", "Рычаг сдернут влево",
                 "Рычаг сдернут вправо", "Оба рычага сдернуты"],
        "FLOOD": ["Затоплен", "Работают все насосы в АНС",
                  "Работают все насосы АНС", "Включены все насосы АНС"],
        "GAS": ["Обнаружен газ"],
        "INTRUSION": ["Обнаружено движение", "Движение вверх", "Движение вниз",
                      "Движение влево", "Движение вправо"],
    }
    open_contact = ["Ручной извещатель", "Тепловой датчик", "Датчик дыма",
                    "Состояние УИР-Р", "Датчик затопления", "КД Дверь",
                    "КД Люк", "КД АВ", "Стекло", "9-секционный люк"]
    sensors = pl.scan_parquet(SENSORS).select(
        pl.col("ид_канала_данных").cast(pl.String).alias("channel_id"),
        pl.col("тип_датчика").alias("sensor_type"),
    ).with_columns(pl.col("sensor_type").replace_strict(mapping, default=None).alias("kind"))
    files = [SOURCE / f"events_{year}.parquet" for year in
             (2019, 2020, 2022, 2023, 2024, 2025, 2026)]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Full original journal is required")
    possible = {value for values in incident_values.values() for value in values}
    possible.add("Не замкнут")
    events = pl.concat([pl.scan_parquet(path) for path in files]).filter(
        pl.col("value").is_in(possible)
    ).select(
        "event_id", "channel_id", "ts", "alarm", "value"
    )
    allowed = pl.lit(False)
    for kind, values in incident_values.items():
        allowed = allowed | ((pl.col("kind") == kind) & pl.col("value").is_in(values))
    allowed = allowed | (pl.col("sensor_type").is_in(open_contact)
                         & (pl.col("value") == "Не замкнут"))
    reference_alarm = pl.col("value").replace_strict(
        alarm_by_value, default=False, return_dtype=pl.Boolean
    )
    target = pl.when(pl.col("kind") == "INTRUSION").then(
        pl.col("alarm")
    ).otherwise(reference_alarm).fill_null(False)
    latest = (events.join(sensors, on="channel_id")
              .filter(allowed & target)
              .sort("channel_id", "ts", "event_id")
              .unique(subset=["channel_id"], keep="last", maintain_order=True)
              .select("channel_id", "ts", "event_id", "kind").collect())
    seed = {channel: [ts.replace(tzinfo=ZoneInfo("Europe/Moscow")).isoformat(),
                      int(event_id), kind]
            for channel, ts, event_id, kind in latest.iter_rows()}
    OUTPUT.write_text(json.dumps({"cutoff": CUTOFF.isoformat(), "latest": seed},
                                 ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"seed channels={len(seed)} source_rows={sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in files)}")


if __name__ == "__main__":
    main()
