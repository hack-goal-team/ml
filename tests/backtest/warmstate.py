"""Состояние канала перед окном загрузки (для прогрева episodes.py).

Без этого первое тревожное событие в окне бэктеста ошибочно читалось бы
как начало нового эпизода, даже если канал был в тревоге уже давно
(пример из независимой проверки: канал 334183, последнее событие до
29.03.2026 — 2026-03-12 14:31:42, тревожное=true).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"
ALL_YEARS = range(2019, 2027)


def _year_rows(dataset_dir: Path, year: int, channel_ids: list[int]) -> pl.DataFrame:
    lf = pl.scan_csv(
        dataset_dir / f"ext-journal-{year}.csv",
        schema_overrides={
            ID_COL: pl.Utf8,
            "дата": pl.Utf8,
            "значение_датчика": pl.Utf8,
        },
        ignore_errors=True,
    )

    return (
        lf.select(ID_COL, "дата", "время", "тревожное", "значение_датчика")
        .filter(pl.col(ID_COL) != ID_COL)
        .with_columns(pl.col(ID_COL).cast(pl.Int64))
        .filter(pl.col(ID_COL).is_in(channel_ids))
        .collect(engine="streaming")
    )


def last_state_before(
    dataset_dir: Path,
    channel_ids: list[int],
    before: datetime,
    years: range = ALL_YEARS,
) -> dict[int, tuple[bool, str]]:
    """Последнее событие каждого канала с ts < before, по всем годам."""
    frames = [_year_rows(dataset_dir, year, channel_ids) for year in years]
    df = pl.concat(frames)

    df = (
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
        .filter(pl.col("ts") < before)
        .sort([ID_COL, "ts"])
        .group_by(ID_COL, maintain_order=True)
        .last()
    )

    return {
        int(row[ID_COL]): (row["тревожное"], row["значение_датчика"])
        for row in df.iter_rows(named=True)
    }
