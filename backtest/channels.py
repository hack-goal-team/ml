"""Точное воспроизведение device-сплита из
retraining/notebooks/2_create_train_test_datasets_V2_MAIN.ipynb (get_device_split).

Универсум id — уникальные ид_канала_данных из журналов 2024-2026
(retraining/notebooks/1_prepare_dataset.ipynb), затем sort + shuffle(seed=69) +
head(90%)/slice(5%)/остаток. Совпадение размеров train/val/test (10366/575/577)
с ноутбуком проверено вручную и задокументировано в docs/backtest/backtest.md.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ID_COL = "ид_канала_данных"
YEARS = (2024, 2025, 2026)
SEED = 69
TRAIN_SIZE = 0.9
VAL_SIZE = 0.05


def _year_ids(dataset_dir: Path, year: int) -> pl.DataFrame:
    # Журналы содержат вклеенные строки-заголовки (см. комментарий в
    # ноутбуке 1) — они не парсятся как Int64 и отбрасываются явно.
    lf = pl.scan_csv(
        dataset_dir / f"ext-journal-{year}.csv",
        schema_overrides={ID_COL: pl.Utf8},
        ignore_errors=True,
    )
    return (
        lf.select(ID_COL)
        .filter(pl.col(ID_COL) != ID_COL)
        .unique()
        .collect(engine="streaming")
        .with_columns(pl.col(ID_COL).cast(pl.Int64))
    )


def all_channel_ids(dataset_dir: Path) -> pl.DataFrame:
    """Уникальные ид каналов из журналов YEARS, отсортированные по возрастанию."""
    frames = [_year_ids(dataset_dir, year) for year in YEARS]
    return pl.concat(frames).unique().sort(ID_COL)


def device_split(dataset_dir: Path) -> pl.DataFrame:
    """Полный train/val/test сплит, один в один как get_device_split."""
    ids = all_channel_ids(dataset_dir).sample(
        fraction=1.0,
        shuffle=True,
        seed=SEED,
    )

    n = ids.height
    n_train = int(n * TRAIN_SIZE)
    n_val = int(n * VAL_SIZE)

    return pl.concat(
        [
            ids.head(n_train).with_columns(pl.lit("train").alias("split")),
            ids.slice(n_train, n_val).with_columns(pl.lit("val").alias("split")),
            ids.slice(n_train + n_val).with_columns(pl.lit("test").alias("split")),
        ]
    )


def test_channel_ids(dataset_dir: Path) -> list[int]:
    split = device_split(dataset_dir)
    return (
        split.filter(pl.col("split") == "test")[ID_COL]
        .sort()
        .to_list()
    )
