"""Честная сверка isAlarm: тревожное журнала против справочник_состояний.

Решение остаётся прежним — берём тревожное журнала (см. episodes.py), но
отчёт должен показывать реальную картину, а не только долю совпадений:
покрытие словаря крошечное, а согласие почти целиком на "не тревога".
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"


def _unambiguous(states: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    agg = states.group_by(group_cols).agg(
        pl.col("тревожное").n_unique().alias("nuniq"),
        pl.col("тревожное").first().alias("тревожное_dict"),
    )
    return agg.filter(pl.col("nuniq") == 1).drop("nuniq")


def _summary(joined: pl.DataFrame) -> pl.DataFrame:
    return (
        joined.rename({"тревожное": "journal"})
        .group_by(["journal", "тревожное_dict"])
        .agg(pl.len().alias("n"))
        .sort(["journal", "тревожное_dict"])
    )


def _mismatches(joined: pl.DataFrame) -> pl.DataFrame:
    return (
        joined.rename({"тревожное": "journal"})
        .filter(pl.col("journal") != pl.col("тревожное_dict"))
        .group_by("значение_датчика")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )


def check(
    events_df: pl.DataFrame,
    states_csv: Path,
    sensors_csv: Path,
) -> dict[str, pl.DataFrame]:
    states = pl.read_csv(states_csv)
    sensors = pl.read_csv(sensors_csv).select(ID_COL, "тип_датчика")
    events = events_df.select(ID_COL, "тревожное", "значение_датчика")

    # Строго: (тип_датчика, название_состояния) однозначны в справочнике.
    strict = _unambiguous(states, ["тип_датчика", "название_состояния"])
    strict_joined = events.join(sensors, on=ID_COL, how="left").join(
        strict,
        left_on=["тип_датчика", "значение_датчика"],
        right_on=["тип_датчика", "название_состояния"],
        how="inner",
    )

    # Шире: только по названию состояния, без учёта типа датчика.
    by_name = _unambiguous(states, ["название_состояния"])
    name_joined = events.join(
        by_name,
        left_on="значение_датчика",
        right_on="название_состояния",
        how="inner",
    )

    return {
        "strict_summary": _summary(strict_joined),
        "strict_coverage": pl.DataFrame(
            {"matched": [strict_joined.height], "total": [events.height]}
        ),
        "by_name_summary": _summary(name_joined),
        "by_name_mismatches": _mismatches(name_joined),
        "by_name_coverage": pl.DataFrame(
            {"matched": [name_joined.height], "total": [events.height]}
        ),
    }
