"""Журналы ext-journal-*.csv и справочники: сплит каналов, события, диспетчеры."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"
SEED = 69
TRAIN_SIZE = 0.9
VAL_SIZE = 0.05


def _scan(dataset_dir: Path, year: int) -> pl.LazyFrame:
    # В журналах есть вклеенные строки-заголовки (см. ноутбук 1), отбрасываем.
    return pl.scan_csv(
        dataset_dir / f"ext-journal-{year}.csv",
        schema_overrides={ID_COL: pl.Utf8, "дата": pl.Utf8, "значение_датчика": pl.Utf8},
        ignore_errors=True,
    ).filter(pl.col(ID_COL) != ID_COL)


def _with_ts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("дата").str.to_date(),
        pl.col("время").str.to_time("%H:%M:%S", strict=False),
        (pl.col("тревожное") == "t").alias("тревожное"),
    ).with_columns(pl.col("дата").dt.combine(pl.col("время")).alias("ts"))


def test_channel_ids(dataset_dir: Path) -> list[int]:
    """Тестовая часть get_device_split из ноутбука 2_create_train_test_datasets_V2_MAIN:
    ид каналов 2024-2026, sort + shuffle(seed=69), train 90% / val 5% / test остаток.
    """
    ids = (
        pl.concat(
            _scan(dataset_dir, year).select(ID_COL).unique()
            .collect(engine="streaming").with_columns(pl.col(ID_COL).cast(pl.Int64))
            for year in (2024, 2025, 2026)
        )
        .unique()
        .sort(ID_COL)
        .sample(fraction=1.0, shuffle=True, seed=SEED)
    )
    n_head = int(ids.height * TRAIN_SIZE) + int(ids.height * VAL_SIZE)
    return ids.slice(n_head)[ID_COL].sort().to_list()


def load_events(
    dataset_dir: Path, channel_ids: list[int], start: date, end: date
) -> pl.DataFrame:
    """События channel_ids за [start, end] из журнала 2026, по каналу и времени."""
    df = (
        _scan(dataset_dir, 2026)
        .select("ид_события", ID_COL, "дата", "время", "тревожное", "значение_датчика")
        .with_columns(pl.col(ID_COL).cast(pl.Int64))
        .filter(pl.col(ID_COL).is_in(channel_ids))
        .filter(pl.col("дата").is_between(pl.lit(start.isoformat()), pl.lit(end.isoformat())))
        .collect(engine="streaming")
    )
    return _with_ts(df).sort([ID_COL, "ts", "ид_события"])


def last_state_before(
    dataset_dir: Path, channel_ids: list[int], before: date
) -> dict[int, bool]:
    """Флаг тревожное последнего события канала до before по всем годам: без него
    первая тревога в окне читалась бы как новый эпизод (канал 334183 в тревоге с 12.03).
    """
    df = pl.concat(
        _scan(dataset_dir, year)
        .select(ID_COL, "дата", "время", "тревожное")
        .with_columns(pl.col(ID_COL).cast(pl.Int64))
        .filter(pl.col(ID_COL).is_in(channel_ids))
        .collect(engine="streaming")
        for year in range(2019, 2027)
    )
    df = (
        _with_ts(df)
        .filter(pl.col("ts") < before)
        .sort([ID_COL, "ts"])
        .group_by(ID_COL, maintain_order=True)
        .last()
    )
    return dict(zip(df[ID_COL].to_list(), df["тревожное"].to_list()))


def channel_dispatchers(sensors_path: Path, objects_path: Path) -> dict[int, int | None]:
    """Диспетчер канала — ближайший предок уровня 2 в справочнике объектов
    (Q&A_2: уровень 2 — «Диспетчер района», 16 объектов).
    """
    objects = pl.read_parquet(objects_path)
    parent = dict(zip(objects["ид_объект"], objects["родитель"]))
    level = dict(zip(objects["ид_объект"], objects["иерархия_уровень"]))

    def dispatcher_of(current: int | None) -> int | None:
        seen = set()
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
    dispatchers: dict[int, int | None], sample_ids: list[int]
) -> dict[int, float]:
    """Множитель «все каналы диспетчера / тестовые каналы диспетчера»."""
    full = Counter(d for d in dispatchers.values() if d is not None)
    sample_set = set(sample_ids)
    sample = Counter(
        d for cid, d in dispatchers.items() if d is not None and cid in sample_set
    )
    return {d: full[d] / count for d, count in sample.items()}
