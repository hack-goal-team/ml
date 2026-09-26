#!/usr/bin/env python3
"""Evaluate the deployed model on the exact test split of a new training run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from scipy.special import ndtr
from sklearn.metrics import average_precision_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    args = parser.parse_args()

    model = CatBoostRegressor()
    model.load_model(str(args.old_model))
    features = model.feature_names_
    if not features or any(not name for name in features):
        raise ValueError("Old model does not contain usable feature names")
    categorical = [features[i] for i in model.get_cat_feature_indices()]
    loss = model.get_all_params()["loss_function"]
    match = re.search(r"(?:^|;)scale=([0-9.eE+-]+)", loss)
    if not match:
        raise ValueError(f"SurvivalAft scale is missing from old model: {loss}")
    scale = float(match.group(1))

    labels = []
    probabilities = []
    test_files = sorted((args.run_dir / "data" / "final_test").glob("chunk_*.parquet"))
    if not test_files:
        raise FileNotFoundError("New run has no test chunks")
    for index, path in enumerate(test_files, 1):
        frame = pl.scan_parquet(path).select(features + ["target_lower", "target_upper"])
        if categorical:
            frame = frame.with_columns(pl.col(categorical).fill_null("null"))
        frame = frame.collect(engine="streaming")
        labels.append(
            ((frame["target_upper"] != -1) & (frame["target_lower"] <= 24.0))
            .cast(pl.Int8).to_numpy()
        )
        pool = Pool(frame.select(features), cat_features=categorical)
        prediction = model.predict(pool)
        probabilities.append(ndtr((np.log(24.0) - prediction) / scale))
        if index % 10 == 0 or index == len(test_files):
            print(f"Old model evaluated on {index}/{len(test_files)} test chunks", flush=True)

    y_test = np.concatenate(labels)
    p_test = np.concatenate(probabilities)
    with (args.run_dir / "survival_optuna" / "scores_final.csv").open(newline="") as file:
        new_scores = next(csv.DictReader(file))
    if len(y_test) != int(new_scores["test_rows"]):
        raise ValueError("Old and new model row counts differ")
    old_pr_auc = float(average_precision_score(y_test, p_test))
    new_pr_auc = float(new_scores["test_pr_auc"])
    result = {
        "test_rows": len(y_test),
        "test_positive_rate": float(y_test.mean()),
        "old_model_test_pr_auc": old_pr_auc,
        "new_model_test_pr_auc": new_pr_auc,
        "delta_pr_auc": new_pr_auc - old_pr_auc,
        "old_model_scale": scale,
        "test_chunks": len(test_files),
    }
    destination = args.run_dir / "survival_optuna" / "comparison_old_vs_new.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
