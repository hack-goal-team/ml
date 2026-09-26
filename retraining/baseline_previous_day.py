#!/usr/bin/env python3
"""Evaluate previous-day alarm rules on the exact final test event rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def score(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    predicted = predictions.astype(bool)
    positive = y_true.astype(bool)
    selected = int(predicted.sum())
    hits = int(np.count_nonzero(predicted & positive))
    return {
        "pr_auc": float(average_precision_score(y_true, predictions)),
        "precision": hits / selected if selected else None,
        "recall": hits / int(positive.sum()) if positive.any() else None,
        "alert_rate": selected / len(y_true),
        "alerts": selected,
        "true_positives": hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    data = args.run_dir / "data"
    test = pl.scan_parquet(str(data / "final_test" / "chunk_*.parquet")).select(
        "ид_события", "ид_канала_данных", "дата", "target_lower", "target_upper"
    )
    test_ids = test.select("ид_канала_данных").unique()
    alarms = (
        pl.scan_parquet(str(data / "ext-journal-*.parquet"))
        .select("ид_канала_данных", "дата", "тревожное")
        .join(test_ids, on="ид_канала_данных", how="semi")
        .with_columns(pl.col("дата").cast(pl.Date))
        .group_by("ид_канала_данных", "дата")
        .agg(pl.col("тревожное").eq("t").sum().alias("alarm_count"))
        .with_columns((pl.col("дата") + pl.duration(days=1)).alias("дата"))
    )
    evaluated = (
        test.join(alarms, on=["ид_канала_данных", "дата"], how="left")
        .select(
            "ид_события",
            ((pl.col("target_upper") != -1) & (pl.col("target_lower") <= 24))
            .cast(pl.Int8)
            .alias("target"),
            pl.col("alarm_count").fill_null(0).alias("alarm_count_yesterday"),
        )
        .collect(engine="streaming")
    )
    y_true = evaluated["target"].to_numpy()
    yesterday = evaluated["alarm_count_yesterday"].to_numpy()
    result = {
        "test_rows": len(y_true),
        "test_positive_rate": float(y_true.mean()),
        "unit": "event row",
        "horizon_hours": 24,
        "rules": {
            "any_alarm_yesterday": score(y_true, (yesterday >= 1).astype(np.int8)),
            "ten_alarms_yesterday": score(y_true, (yesterday >= 10).astype(np.int8)),
        },
    }
    destination = args.run_dir / "survival_optuna" / "baseline_previous_day.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
