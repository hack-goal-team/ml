"""Загрузка событий тестовых каналов за окно бэктеста из ext-journal CSV."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"
RAW_COLUMNS = (
    "ид_события",
    ID_COL,
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
)


def load_events(
    dataset_dir: Path,
    channel_ids: list[int],
    start: date,
    end: date,
    year: int = 2026,
) -> pl.DataFrame:
    """События channel_ids за [start, end], отсортированные по каналу и времени."""
    lf = pl.scan_csv(
        dataset_dir / f"ext-journal-{year}.csv",
        schema_overrides={ID_COL: pl.Utf8, "дата": pl.Utf8},
        ignore_errors=True,
    )

    df = (
        lf.select(*RAW_COLUMNS)
        .filter(pl.col(ID_COL) != ID_COL)
        .with_columns(pl.col(ID_COL).cast(pl.Int64))
        .filter(pl.col(ID_COL).is_in(channel_ids))
        .filter(
            pl.col("дата").is_between(
                pl.lit(start.isoformat()),
                pl.lit(end.isoformat()),
            )
        )
        .collect(engine="streaming")
    )

    return (
        df.with_columns(
            pl.col("дата").str.to_date(),
            pl.col("время").str.to_time("%H:%M:%S", strict=False),
            (pl.col("тревожное") == "t").alias("тревожное"),
        )
        .with_columns(
            pl.datetime(
                pl.col("дата").dt.year(),
                pl.col("дата").dt.month(),
                pl.col("дата").dt.day(),
                pl.col("время").dt.hour(),
                pl.col("время").dt.minute(),
                pl.col("время").dt.second(),
            ).alias("ts")
        )
        .sort([ID_COL, "ts", "ид_события"])
    )
