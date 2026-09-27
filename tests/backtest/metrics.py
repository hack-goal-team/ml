"""Precision/recall, алерты на смену диспетчера, калибровка, бутстрэп CI."""
from __future__ import annotations

import numpy as np
import polars as pl

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4
N_BOOTSTRAP = 2000


def _binary_metrics(
    df: pl.DataFrame, predicted: pl.Series, dispatcher_scale: dict[int, float]
) -> dict[str, float]:
    df = df.with_columns(predicted.alias("__pred"))
    tp = df.filter(pl.col("__pred") & pl.col("outcome_24h")).height
    fp = df.filter(pl.col("__pred") & ~pl.col("outcome_24h")).height
    fn = df.filter(~pl.col("__pred") & pl.col("outcome_24h")).height

    by_dispatcher_shift = (
        df.filter(pl.col("dispatcher_id").is_not_null())
        .group_by(["dispatcher_id", "t0"])
        .agg(pl.col("__pred").sum().alias("alerts"))
        .with_columns(
            pl.col("dispatcher_id")
            .replace_strict(dispatcher_scale, default=1.0, return_dtype=pl.Float64)
            .alias("scale")
        )
    )
    hours = df.filter(pl.col("__pred") & pl.col("outcome_24h"))["hours_to_event"].drop_nulls()

    return {
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": df.height,
        # На тестовых каналах (~5% парка) и пересчитанное на весь парк диспетчера.
        "alerts_per_shift_sample": by_dispatcher_shift["alerts"].mean(),
        "alerts_per_shift_fleet": (by_dispatcher_shift["alerts"] * by_dispatcher_shift["scale"]).mean(),
        "median_hours_to_event": hours.median() if hours.len() else float("nan"),
    }


def _method_predicates(df: pl.DataFrame) -> dict[str, pl.Series]:
    return {
        "model_fresh_t0_high_0.7": df["probability"] >= HIGH_THRESHOLD,
        "model_fresh_t0_medium_0.4": df["probability"] >= MEDIUM_THRESHOLD,
        "model_prodstyle_high_0.7": df["prodstyle_valid"] & (df["prodstyle_probability"] >= HIGH_THRESHOLD),
        "model_prodstyle_medium_0.4": df["prodstyle_valid"] & (df["prodstyle_probability"] >= MEDIUM_THRESHOLD),
        "baseline_alarm_yesterday": df["baseline_alarm_yesterday"],
        "baseline_10plus_yesterday": df["baseline_10plus_yesterday"],
    }


def build_comparison_table(df: pl.DataFrame, dispatcher_scale: dict[int, float]) -> pl.DataFrame:
    slices = {"all_channels": df, "early_warning_6h": df.filter(pl.col("early_warning_ok"))}
    return pl.DataFrame(
        [
            {"slice": slice_name, "method": method, **_binary_metrics(sdf, predicate, dispatcher_scale)}
            for slice_name, sdf in slices.items()
            for method, predicate in _method_predicates(sdf).items()
        ]
    )


def calibration_table(df: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    breaks = [i / bins for i in range(1, bins)]
    labels = [f"[{i / bins:.1f}, {(i + 1) / bins:.1f})" for i in range(bins)]
    return (
        df.with_columns(
            pl.col("probability").cut(breaks, labels=labels, left_closed=True).alias("probability_bin")
        )
        .group_by("probability_bin", maintain_order=True)
        .agg(
            pl.len().alias("n"),
            pl.col("outcome_24h").mean().alias("actual_rate"),
            pl.col("probability").mean().alias("mean_probability"),
        )
        .sort("probability_bin")
    )


def _per_channel_counts(df: pl.DataFrame, predicted: pl.Series) -> list[np.ndarray]:
    counts = (
        df.with_columns(predicted.alias("__pred"))
        .group_by("channel_id")
        .agg(
            (pl.col("__pred") & pl.col("outcome_24h")).sum().alias("tp"),
            (pl.col("__pred") & ~pl.col("outcome_24h")).sum().alias("fp"),
            (~pl.col("__pred") & pl.col("outcome_24h")).sum().alias("fn"),
        )
        .sort("channel_id")
    )
    return [counts[c].to_numpy() for c in ("tp", "fp", "fn")]


def bootstrap_diff_ci(
    df: pl.DataFrame, predicted_a: pl.Series, predicted_b: pl.Series
) -> dict[str, float]:
    """95% CI разниц precision и recall a - b. Бутстрэп по каналам, не по строкам:
    смены одного канала коррелированы, построчный занизил бы дисперсию.
    """
    tp_a, fp_a, fn_a = _per_channel_counts(df, predicted_a)
    tp_b, fp_b, fn_b = _per_channel_counts(df, predicted_b)

    rng = np.random.default_rng(0)
    n = tp_a.shape[0]
    diffs = {"precision_diff": np.empty(N_BOOTSTRAP), "recall_diff": np.empty(N_BOOTSTRAP)}

    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        a_tp, a_fp, a_fn = tp_a[idx].sum(), fp_a[idx].sum(), fn_a[idx].sum()
        b_tp, b_fp, b_fn = tp_b[idx].sum(), fp_b[idx].sum(), fn_b[idx].sum()
        diffs["precision_diff"][i] = (a_tp / (a_tp + a_fp) if (a_tp + a_fp) else 0.0) - (
            b_tp / (b_tp + b_fp) if (b_tp + b_fp) else 0.0
        )
        diffs["recall_diff"][i] = (a_tp / (a_tp + a_fn) if (a_tp + a_fn) else 0.0) - (
            b_tp / (b_tp + b_fn) if (b_tp + b_fn) else 0.0
        )

    result = {}
    for name, values in diffs.items():
        result[name] = float(np.mean(values))
        result[f"{name}_ci_low"] = float(np.percentile(values, 2.5))
        result[f"{name}_ci_high"] = float(np.percentile(values, 97.5))
    return result


def build_bootstrap_table(df: pl.DataFrame) -> pl.DataFrame:
    """CI каждого варианта модели против бейзлайна «тревога вчера»."""
    predicates = _method_predicates(df)
    baseline = predicates["baseline_alarm_yesterday"]
    return pl.DataFrame(
        [
            {"method": method, "vs": "baseline_alarm_yesterday", **bootstrap_diff_ci(df, predicate, baseline)}
            for method, predicate in predicates.items()
            if method.startswith("model_")
        ]
    )
