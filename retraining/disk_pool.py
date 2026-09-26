"""Build numeric CatBoost pools from Parquet without collecting every row in Python."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from catboost.utils import quantize


def quantized_pool(
    frame: pl.LazyFrame,
    features: list[str],
    target_columns: list[str],
    output_dir: Path,
    split: str,
    input_borders: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"{split}.tsv"
    description = output_dir / "survival.cd"
    description.write_text("0\tLabel\n1\tLabel\n", encoding="utf-8")

    schema = frame.select(features).collect_schema()
    non_numeric = [name for name in features if not schema[name].is_numeric()]
    if non_numeric:
        raise ValueError(f"Disk quantization requires numeric features: {non_numeric}")

    frame.select(target_columns + features).sink_csv(
        table,
        separator="\t",
        null_value="nan",
    )
    options = {
        "column_description": str(description),
        "has_header": True,
    }
    if input_borders is not None:
        options["input_borders"] = str(input_borders)
    pool = quantize(str(table), **options)
    table.unlink()
    return pool


def binary_target(frame: pl.LazyFrame, horizon: float) -> np.ndarray:
    return (
        frame.select(
            (
                (pl.col("target_upper") != -1)
                & (pl.col("target_lower") <= horizon)
            ).cast(pl.Int8)
        )
        .collect(engine="streaming")
        .to_series()
        .to_numpy()
    )
