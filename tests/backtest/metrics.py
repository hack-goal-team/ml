"""Метрики бэктеста: precision/recall/алерты на смену/медиана часов,
калибровка и кластерный (по каналам) бутстрэп доверительных интервалов.
"""
from __future__ import annotations

import numpy as np
import polars as pl

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


def _binary_metrics(
    df: pl.DataFrame,
    predicted: pl.Series,
    dispatcher_scale: dict[int, float] | None = None,
) -> dict[str, float]:
    df = df.with_columns(predicted.alias("__pred"))

    tp = df.filter(pl.col("__pred") & pl.col("outcome_24h")).height
    fp = df.filter(pl.col("__pred") & ~pl.col("outcome_24h")).height
    fn = df.filter(~pl.col("__pred") & pl.col("outcome_24h")).height

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    by_dispatcher_shift = (
        df.filter(pl.col("dispatcher_id").is_not_null())
        .group_by(["dispatcher_id", "t0"])
        .agg(pl.col("__pred").sum().alias("alerts"))
    )

    alerts_per_shift_sample = (
        by_dispatcher_shift["alerts"].mean() if by_dispatcher_shift.height else float("nan")
    )

    alerts_per_shift_fleet = float("nan")
    if dispatcher_scale and by_dispatcher_shift.height:
        scaled = by_dispatcher_shift.with_columns(
            pl.col("dispatcher_id")
            .replace_strict(dispatcher_scale, default=1.0, return_dtype=pl.Float64)
            .alias("scale")
        ).with_columns((pl.col("alerts") * pl.col("scale")).alias("alerts_scaled"))
        alerts_per_shift_fleet = scaled["alerts_scaled"].mean()

    hours = df.filter(pl.col("__pred") & pl.col("outcome_24h"))["hours_to_event"].drop_nulls()
    median_hours = hours.median() if hours.len() else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": df.height,
        # На тестовой выборке каналов (~5% парка) и пересчитанное на весь
        # парк диспетчера (см. backtest.dispatch.dispatcher_scale).
        "alerts_per_shift_sample": alerts_per_shift_sample,
        "alerts_per_shift_fleet": alerts_per_shift_fleet,
        "median_hours_to_event": median_hours,
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


def build_comparison_table(
    df: pl.DataFrame,
    dispatcher_scale: dict[int, float] | None = None,
) -> pl.DataFrame:
    slices = {
        "all_channels": df,
        "early_warning_6h": df.filter(pl.col("early_warning_ok")),
    }

    rows = []
    for slice_name, sdf in slices.items():
        for method_name, predicate in _method_predicates(sdf).items():
            metrics = _binary_metrics(sdf, predicate, dispatcher_scale)
            rows.append({"slice": slice_name, "method": method_name, **metrics})

    return pl.DataFrame(rows)


def calibration_table(
    df: pl.DataFrame,
    probability_col: str = "probability",
    bins: int = 10,
) -> pl.DataFrame:
    breaks = [i / bins for i in range(1, bins)]
    labels = [f"[{i / bins:.1f}, {(i + 1) / bins:.1f})" for i in range(bins)]

    return (
        df.with_columns(
            pl.col(probability_col)
            .cut(breaks, labels=labels, left_closed=True)
            .alias("probability_bin")
        )
        .group_by("probability_bin", maintain_order=True)
        .agg(
            pl.len().alias("n"),
            pl.col("outcome_24h").mean().alias("actual_rate"),
            pl.col(probability_col).mean().alias("mean_probability"),
        )
        .sort("probability_bin")
    )


def _per_channel_counts(df: pl.DataFrame, predicted: pl.Series) -> pl.DataFrame:
    return (
        df.with_columns(predicted.alias("__pred"))
        .group_by("channel_id")
        .agg(
            (pl.col("__pred") & pl.col("outcome_24h")).sum().alias("tp"),
            (pl.col("__pred") & ~pl.col("outcome_24h")).sum().alias("fp"),
            (~pl.col("__pred") & pl.col("outcome_24h")).sum().alias("fn"),
        )
        .sort("channel_id")
    )


def bootstrap_diff_ci(
    df: pl.DataFrame,
    predicted_a: pl.Series,
    predicted_b: pl.Series,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    """95% CI разницы (precision_a - precision_b) и (recall_a - recall_b).

    Кластерный бутстрэп по каналам (не по строкам): ресемплируем каналы с
    возвратом и суммируем их tp/fp/fn — строки одного канала коррелированы
    (одни и те же 182 смены), построчный бутстрэп занизил бы дисперсию.
    """
    counts_a = _per_channel_counts(df, predicted_a)
    counts_b = _per_channel_counts(df, predicted_b)

    tp_a, fp_a, fn_a = (counts_a[c].to_numpy() for c in ("tp", "fp", "fn"))
    tp_b, fp_b, fn_b = (counts_b[c].to_numpy() for c in ("tp", "fp", "fn"))

    rng = np.random.default_rng(seed)
    n = tp_a.shape[0]
    precision_diffs = np.empty(n_boot)
    recall_diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a_tp, a_fp, a_fn = tp_a[idx].sum(), fp_a[idx].sum(), fn_a[idx].sum()
        b_tp, b_fp, b_fn = tp_b[idx].sum(), fp_b[idx].sum(), fn_b[idx].sum()

        precision_a = a_tp / (a_tp + a_fp) if (a_tp + a_fp) else 0.0
        precision_b = b_tp / (b_tp + b_fp) if (b_tp + b_fp) else 0.0
        recall_a = a_tp / (a_tp + a_fn) if (a_tp + a_fn) else 0.0
        recall_b = b_tp / (b_tp + b_fn) if (b_tp + b_fn) else 0.0

        precision_diffs[i] = precision_a - precision_b
        recall_diffs[i] = recall_a - recall_b

    return {
        "precision_diff": float(np.mean(precision_diffs)),
        "precision_diff_ci_low": float(np.percentile(precision_diffs, 2.5)),
        "precision_diff_ci_high": float(np.percentile(precision_diffs, 97.5)),
        "recall_diff": float(np.mean(recall_diffs)),
        "recall_diff_ci_low": float(np.percentile(recall_diffs, 2.5)),
        "recall_diff_ci_high": float(np.percentile(recall_diffs, 97.5)),
    }


def build_bootstrap_table(df: pl.DataFrame) -> pl.DataFrame:
    """CI для моделей (обоих вариантов) против бейзлайна «тревога вчера»."""
    predicates = _method_predicates(df)
    baseline = predicates["baseline_alarm_yesterday"]

    model_methods = [
        "model_fresh_t0_high_0.7",
        "model_fresh_t0_medium_0.4",
        "model_prodstyle_high_0.7",
        "model_prodstyle_medium_0.4",
    ]

    rows = []
    for method in model_methods:
        ci = bootstrap_diff_ci(df, predicates[method], baseline)
        rows.append({"method": method, "vs": "baseline_alarm_yesterday", **ci})

    return pl.DataFrame(rows)
