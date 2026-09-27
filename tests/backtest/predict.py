"""Реплей событий через app.PredictionService в двух вариантах: прогноз ровно
на смене t0 и «как на стенде» — последний прогноз по событию до t0.
"""
from __future__ import annotations

import bisect
from datetime import datetime

import polars as pl

from app import PredictionService
from tests.backtest.data import ID_COL

# Стенд пишет прогноз на каждом событии с horizon_until = ts + 30ч и удаляет
# просроченные (inference/runner.py, predictions.delete_expired).
PROD_HORIZON_HOURS = 30.0


def _channel_events(events_df: pl.DataFrame) -> dict[int, list[tuple[datetime, str]]]:
    return {
        int(channel_id): list(zip(group["ts"].to_list(), group["значение_датчика"].to_list()))
        for (channel_id,), group in events_df.partition_by(
            ID_COL, as_dict=True, maintain_order=True
        ).items()
    }


def predict_shifts(
    service: PredictionService,
    events_df: pl.DataFrame,
    channel_ids: list[int],
    shift_times: list[datetime],
    horizon_hours: float,
) -> list[tuple[int, datetime, float, bool]]:
    """(канал, t0, вероятность, history_complete) по событиям с ts < t0."""
    channel_events = _channel_events(events_df)
    rows, keys = [], []

    for channel_id in channel_ids:
        events = channel_events.get(channel_id, [])
        ei = 0
        for t0 in shift_times:
            while ei < len(events) and events[ei][0] < t0:
                service.store.update(channel_id, *events[ei])
                ei += 1
            rows.append(service._build_feature_row(channel_id, t0))
            keys.append((channel_id, t0, service.store.history_complete(channel_id, t0)))

    raw = service.model.predict(rows)
    return [
        (channel_id, t0, service._probability_from_raw(float(r), horizon_hours), complete)
        for (channel_id, t0, complete), r in zip(keys, raw)
    ]


def predict_last_event(
    service: PredictionService,
    events_df: pl.DataFrame,
    channel_ids: list[int],
    shift_times: list[datetime],
) -> dict[tuple[int, datetime], tuple[float | None, float | None]]:
    """{(канал, t0): (вероятность, возраст в часах)} последнего прогноза по
    событию до t0; (None, None), если событий до t0 не было.
    """
    channel_events = _channel_events(events_df)
    result: dict[tuple[int, datetime], tuple[float | None, float | None]] = {}
    rows, row_meta = [], []

    for channel_id in channel_ids:
        events = channel_events.get(channel_id, [])
        ts = [event_ts for event_ts, _ in events]

        # Смены по индексу последнего события со ts < t0.
        due: dict[int, list[datetime]] = {}
        for t0 in shift_times:
            due.setdefault(bisect.bisect_left(ts, t0) - 1, []).append(t0)
        for t0 in due.get(-1, []):
            result[(channel_id, t0)] = (None, None)

        for idx, (event_ts, value) in enumerate(events):
            service.store.update(channel_id, event_ts, value)
            t0s = due.get(idx)
            if not t0s:
                continue
            # Следующее событие с тем же ts: прогноз стенд даст после него.
            if idx + 1 < len(ts) and ts[idx + 1] == event_ts:
                due.setdefault(idx + 1, []).extend(t0s)
                continue
            rows.append(service._build_feature_row(channel_id, event_ts))
            row_meta.append((channel_id, event_ts, t0s))

    for (channel_id, event_ts, t0s), raw in zip(row_meta, service.model.predict(rows)):
        probability = service._probability_from_raw(float(raw), PROD_HORIZON_HOURS)
        for t0 in t0s:
            result[(channel_id, t0)] = (probability, (t0 - event_ts).total_seconds() / 3600)

    return result
