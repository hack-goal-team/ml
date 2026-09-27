"""Оффлайн-бэктест HACK-174, см. tests/backtest/backtest.md.

    HACK174_DATASET_DIR=/path/to/Sources/dataset python -m tests.backtest.run
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from tests.backtest import data, episodes, metrics, predict
from tests.backtest.service import build_offline_service

REPO_ROOT = Path(__file__).resolve().parents[2]
HORIZON = timedelta(hours=24)
SHIFT_HOURS = (8, 20)
WARMUP_PAD_DAYS = 3


def _build_rows(
    predictions: list[tuple[int, datetime, float, bool]],
    prodstyle: dict[tuple[int, datetime], tuple[float | None, float | None]],
    timelines: dict[int, episodes.ChannelTimeline],
    dispatchers: dict[int, int | None],
) -> pl.DataFrame:
    rows = []
    for channel_id, t0, probability, history_complete in predictions:
        timeline = timelines[channel_id]
        yesterday_alarms = episodes.alarms_in_window(timeline, t0 - timedelta(hours=24), t0)
        # Спокойный канал: сейчас в норме и без тревог последние 6 ч.
        early_warning_ok = (
            episodes.state_at(timeline, t0) is not True
            and episodes.alarms_in_window(timeline, t0 - timedelta(hours=6), t0) == 0
        )
        episode_start = episodes.first_episode_after(timeline, t0, HORIZON)
        prod_probability, prod_age = prodstyle[(channel_id, t0)]

        rows.append(
            {
                "channel_id": channel_id,
                "t0": t0,
                "probability": probability,
                "history_complete": history_complete,
                "prodstyle_probability": prod_probability,
                "prodstyle_age_hours": prod_age,
                "prodstyle_valid": prod_age is not None and prod_age < predict.PROD_HORIZON_HOURS,
                "baseline_alarm_yesterday": yesterday_alarms > 0,
                "baseline_10plus_yesterday": yesterday_alarms >= 10,
                "early_warning_ok": early_warning_ok,
                "outcome_24h": episode_start is not None,
                "hours_to_event": (
                    (episode_start - t0).total_seconds() / 3600 if episode_start is not None else None
                ),
                "dispatcher_id": dispatchers.get(channel_id),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("HACK174_DATASET_DIR"))
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=REPO_ROOT / "retraining/data/reference/open-meteo-55.75N37.63E140m.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 30))
    args = parser.parse_args()
    if args.dataset_dir is None:
        parser.error("--dataset-dir or HACK174_DATASET_DIR is required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    test_ids = data.test_channel_ids(args.dataset_dir)
    pl.DataFrame({data.ID_COL: test_ids}).write_csv(out / "test_channels.csv")

    load_start = args.start - timedelta(days=WARMUP_PAD_DAYS)
    events_df = data.load_events(
        args.dataset_dir, test_ids, load_start, args.end + timedelta(days=1, hours=1)
    )
    seed_states = data.last_state_before(args.dataset_dir, test_ids, load_start)
    service = build_offline_service(REPO_ROOT, args.weather_csv)

    have_events = set(events_df[data.ID_COL].unique().to_list())
    usable_ids = [cid for cid in test_ids if cid in have_events and cid in service.metadata]

    # Смены, у которых окно t0 + 24ч целиком внутри периода: последняя — 29.06 20:00.
    period_start = datetime.combine(args.start, datetime.min.time())
    period_end = datetime.combine(args.end, datetime.min.time()) + timedelta(days=1)
    shift_times = [
        period_start + timedelta(days=d, hours=h)
        for d in range((args.end - args.start).days + 1)
        for h in SHIFT_HOURS
        if period_start + timedelta(days=d, hours=h) + HORIZON <= period_end
    ]

    predictions = predict.predict_shifts(
        service, events_df, usable_ids, shift_times, HORIZON.total_seconds() / 3600
    )
    # Store уже продвинут до последней смены: без сброса события второго
    # прохода app.OnlineFeatureStore.update счёл бы out-of-order.
    service.store.states.clear()
    prodstyle = predict.predict_last_event(service, events_df, usable_ids, shift_times)

    timelines = episodes.build_timelines(events_df, seed_states)
    dispatchers = data.channel_dispatchers(
        service.config.sensors_metadata_path, service.config.objects_metadata_path
    )
    df = _build_rows(predictions, prodstyle, timelines, dispatchers)
    df.write_csv(out / "predictions.csv")

    episodes.episode_value_breakdown(timelines, period_start, period_end).write_csv(
        out / "episode_value_breakdown.csv"
    )
    comparison = metrics.build_comparison_table(df, data.dispatcher_scale(dispatchers, usable_ids))
    comparison.write_csv(out / "comparison_table.csv", float_precision=6)
    metrics.calibration_table(df).write_csv(out / "calibration.csv")
    metrics.build_bootstrap_table(df).write_csv(out / "bootstrap_ci.csv")
    print(comparison)


if __name__ == "__main__":
    main()
