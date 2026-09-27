"""Диспетчер канала = ближайший предок уровня 2 в справочнике объектов.

Sources/Q&A/Q&A_2.txt (Maxim, 9/18): иерархия_уровень 1 — единственный
корневой "district", 2 — "Диспетчер района" (16 объектов controlHouse/
guardObject), 3 — их дочерние подобъекты. Каналы висят на объектах любого
уровня; поднимаемся по "родитель" до первого уровня 2.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"


def channel_dispatchers(
    sensors_path: Path,
    objects_path: Path,
) -> dict[int, int | None]:
    objects = pl.read_parquet(objects_path)
    parent = dict(zip(objects["ид_объект"], objects["родитель"]))
    level = dict(zip(objects["ид_объект"], objects["иерархия_уровень"]))

    def dispatcher_of(object_id: int) -> int | None:
        current, seen = object_id, set()

        while current is not None and current not in seen:
            if level.get(current) == 2:
                return int(current)

            seen.add(current)
            current = parent.get(current)

        return None

    sensors = pl.read_parquet(sensors_path, columns=[ID_COL, "ид_объект"])

    return {
        int(channel_id): dispatcher_of(int(object_id))
        for channel_id, object_id in zip(sensors[ID_COL], sensors["ид_объект"])
    }


def dispatcher_scale(
    all_dispatchers: dict[int, int | None],
    sample_channel_ids: list[int],
) -> dict[int, float]:
    """Множитель «весь парк диспетчера / тестовые каналы диспетчера».

    Тестовые каналы — случайные ~5% каналов (по устройству, не по
    диспетчеру), поэтому экстраполяция на весь парк предполагает
    однородность каналов внутри диспетчера; при малой тестовой выборке на
    диспетчера множитель может быть неустойчивым (см. допущения в
    backtest.md).
    """
    full_counts = Counter(d for d in all_dispatchers.values() if d is not None)
    sample_set = set(sample_channel_ids)
    sample_counts = Counter(
        d for cid, d in all_dispatchers.items() if d is not None and cid in sample_set
    )

    return {
        dispatcher: full_counts[dispatcher] / count
        for dispatcher, count in sample_counts.items()
        if count > 0
    }
