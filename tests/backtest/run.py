"""Оффлайн-бэктест HACK-174: одна команда, весь конвейер.

    python -m tests.backtest.run --dataset-dir /path/to/Sources/dataset

Путь к данным обязателен: через --dataset-dir или переменную окружения
HACK174_DATASET_DIR (см. backtest/README.md). Читает журналы событий и
справочники, поднимает боевую app.PredictionService на data/best_model.cbm
(та же модель, что на стенде), считает вероятность тревоги на смены
апрель-июнь 2026 по тестовым каналам сплита в двух вариантах — «прогноз
ровно на границе смены» и «то, что стенд реально показывал по последнему
событию» — и сравнивает с двумя бейзлайнами. Результат — tests/backtest/results/*.csv
(predictions.csv не коммитится, см. .gitignore) и tests/backtest/backtest.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.backtest import channels, data, episodes, metrics, prodstyle  # noqa: E402
from tests.backtest.dispatch import channel_dispatchers, dispatcher_scale  # noqa: E402
from tests.backtest.predict import predict_shifts  # noqa: E402
from tests.backtest.service import build_offline_service  # noqa: E402
from tests.backtest.warmstate import last_state_before  # noqa: E402

HORIZON_HOURS = 24.0
SHIFT_HOURS = (8, 20)
WARMUP_PAD_DAYS = 3
OUTCOME_PAD_DAYS = 1


def _shift_times(start: date, end: date) -> list[datetime]:
    times, day = [], start
    while day <= end:
        times.extend(datetime(day.year, day.month, day.day, h) for h in SHIFT_HOURS)
        day += timedelta(days=1)
    return times


def _valid_shift_times(start: date, end: date, horizon_hours: float) -> list[datetime]:
    """Смены, у которых t0+horizon целиком внутри [start, end] (см. п.6
    сноски проверяющего): последняя валидная смена — 29.06 20:00, не 30.06.
    """
    period_end = datetime(end.year, end.month, end.day) + timedelta(days=1)
    horizon = timedelta(hours=horizon_hours)
    return [t0 for t0 in _shift_times(start, end) if t0 + horizon <= period_end]


def _build_rows(
    predictions: list,
    prodstyle_by_key: dict[tuple[int, datetime], prodstyle.ProdPrediction],
    timelines: dict[int, episodes.ChannelTimeline],
    dispatchers: dict[int, int | None],
) -> pl.DataFrame:
    horizon = timedelta(hours=HORIZON_HOURS)
    rows = []

    for pred in predictions:
        timeline = timelines[pred.channel_id]
        yesterday_start = pred.t0 - timedelta(hours=24)
        yesterday_alarms = episodes.alarms_in_window(timeline, yesterday_start, pred.t0)

        # "Спокойный" канал: сейчас в норме И без тревог последние 6ч —
        # без первого условия канал, который тревожит уже давно без новых
        # событий, ошибочно считался бы спокойным (см. warmstate.py).
        quiet_no_recent_alarm = (
            episodes.alarms_in_window(timeline, pred.t0 - timedelta(hours=6), pred.t0) == 0
        )
        currently_normal = episodes.state_at(timeline, pred.t0) is not True
        early_warning_ok = currently_normal and quiet_no_recent_alarm

        episode_start = episodes.first_episode_after(timeline, pred.t0, horizon)
        prod = prodstyle_by_key.get((pred.channel_id, pred.t0))

        rows.append(
            {
                "channel_id": pred.channel_id,
                "t0": pred.t0,
                "probability": pred.probability,
                "history_complete": pred.history_complete,
                "prodstyle_probability": prod.probability if prod else None,
                "prodstyle_age_hours": prod.age_hours if prod else None,
                "prodstyle_valid": bool(prod and prod.valid),
                "baseline_alarm_yesterday": yesterday_alarms > 0,
                "baseline_10plus_yesterday": yesterday_alarms >= 10,
                "early_warning_ok": early_warning_ok,
                "outcome_24h": episode_start is not None,
                "hours_to_event": (
                    (episode_start - pred.t0).total_seconds() / 3600
                    if episode_start is not None
                    else None
                ),
                "dispatcher_id": dispatchers.get(pred.channel_id),
            }
        )

    return pl.DataFrame(rows, infer_schema_length=None)


def _require_dataset_dir(value: str | None) -> Path:
    resolved = value or os.environ.get("HACK174_DATASET_DIR")
    if not resolved:
        raise SystemExit(
            "--dataset-dir is required (or set HACK174_DATASET_DIR): "
            "каталог с ext-journal-*.csv и справочниками, см. backtest/README.md"
        )
    return Path(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=REPO_ROOT / "retraining/data/reference/open-meteo-55.75N37.63E140m.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    # Своя temp-директория на прогон: устаревший metadata-кеш/state не
    # должен подмешаться (тот же приём, что и в inference/runner.py).
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="hack174-backtest-")),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 30))
    args = parser.parse_args()
    dataset_dir = _require_dataset_dir(args.dataset_dir)

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1/7 test-channel split (notebooks/2, seed=69)...")
    test_ids = channels.test_channel_ids(dataset_dir)
    pl.DataFrame({"ид_канала_данных": test_ids}).write_csv(
        args.output_dir / "test_channels.csv"
    )
    print(f"    test channels: {len(test_ids)}")

    load_start = args.start - timedelta(days=WARMUP_PAD_DAYS)
    load_end = args.end + timedelta(days=OUTCOME_PAD_DAYS, hours=1)
    print(f"2/7 loading events [{load_start}, {load_end}] for test channels...")
    events_df = data.load_events(dataset_dir, test_ids, load_start, load_end)
    print(f"    events: {events_df.height}")

    print("3/7 warming episode state from before the load window...")
    seed_states = last_state_before(dataset_dir, test_ids, load_start)

    print("4/7 building service (app.PredictionService, offline weather)...")
    service = build_offline_service(REPO_ROOT, args.weather_csv, args.runtime_dir)

    have_events = set(events_df["ид_канала_данных"].unique().to_list())
    usable_ids = [
        cid for cid in test_ids if cid in have_events and cid in service.metadata
    ]
    print(
        f"    usable channels: {len(usable_ids)} / {len(test_ids)} "
        f"(no events: {len(set(test_ids) - have_events)}, "
        f"no metadata: {len(set(test_ids) - set(service.metadata))})"
    )

    shift_times = _valid_shift_times(args.start, args.end, HORIZON_HOURS)
    print(
        f"5/7 predicting {len(usable_ids)} channels x {len(shift_times)} shifts "
        f"(fresh-t0 + prodstyle)..."
    )
    predictions = predict_shifts(
        service, events_df, usable_ids, shift_times, HORIZON_HOURS
    )
    # Второй проход по своим таймлайнам: store уже продвинут до последней
    # смены предыдущим проходом, иначе события "из прошлого" будут
    # выглядеть как out-of-order (app.OnlineFeatureStore.update).
    service.store.states.clear()
    prod_predictions = prodstyle.predict_last_event(
        service, events_df, usable_ids, shift_times
    )
    prodstyle_by_key = {(p.channel_id, p.t0): p for p in prod_predictions}

    print("6/7 episodes, baselines, dispatcher grouping...")
    timelines = episodes.build_timelines(events_df, seed_states)
    dispatchers_all = channel_dispatchers(
        service.config.sensors_metadata_path,
        service.config.objects_metadata_path,
    )
    scale = dispatcher_scale(dispatchers_all, usable_ids)
    df = _build_rows(predictions, prodstyle_by_key, timelines, dispatchers_all)
    df.write_csv(args.output_dir / "predictions.csv")

    period_end = datetime(args.end.year, args.end.month, args.end.day) + timedelta(days=1)
    breakdown = episodes.episode_value_breakdown(
        timelines, datetime(args.start.year, args.start.month, args.start.day), period_end
    )
    breakdown.write_csv(args.output_dir / "episode_value_breakdown.csv")

    print("7/7 metrics, bootstrap CI, report...")
    comparison = metrics.build_comparison_table(df, dispatcher_scale=scale)
    comparison.write_csv(float_precision=6, file=args.output_dir / "comparison_table.csv")
    calibration = metrics.calibration_table(df)
    calibration.write_csv(args.output_dir / "calibration.csv")
    bootstrap = metrics.build_bootstrap_table(df)
    bootstrap.write_csv(args.output_dir / "bootstrap_ci.csv")

    elapsed = time.perf_counter() - started
    print(f"done in {elapsed:.1f}s")
    print(comparison)
    print(calibration)
    print(bootstrap)
    print(breakdown.head(10))


if __name__ == "__main__":
    main()
