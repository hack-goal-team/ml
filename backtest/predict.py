"""Реплей событий через app.PredictionService + батч model.predict.

Признаки на чек-поинте t0 строятся строго по событиям с ts < t0 (см.
app.PredictionService._build_feature_row): перед каждым чек-поинтом
докручиваем store.update по всем более ранним событиям канала, затем
строим строку фичей на месте прогноза, ничего не переписывая в app.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from app import PredictionService

ID_COL = "ид_канала_данных"


@dataclass(slots=True)
class ShiftPrediction:
    channel_id: int
    t0: datetime
    probability: float
    history_complete: bool


def _channel_events(
    events_df: pl.DataFrame,
) -> dict[int, list[tuple[datetime, str]]]:
    parts = events_df.partition_by(ID_COL, as_dict=True, maintain_order=True)

    return {
        int(channel_id): list(
            zip(group["ts"].to_list(), group["значение_датчика"].to_list())
        )
        for (channel_id,), group in parts.items()
    }


def predict_shifts(
    service: PredictionService,
    events_df: pl.DataFrame,
    channel_ids: list[int],
    shift_times: list[datetime],
    horizon_hours: float,
) -> list[ShiftPrediction]:
    channel_events = _channel_events(events_df)

    rows: list[list] = []
    meta: list[tuple[int, datetime, bool]] = []

    for channel_id in channel_ids:
        events = channel_events.get(channel_id, [])
        ei, n = 0, len(events)

        for t0 in shift_times:
            while ei < n and events[ei][0] < t0:
                ts, value = events[ei]
                service.store.update(channel_id, ts, value)
                ei += 1

            row = service._build_feature_row(channel_id, t0)
            complete = service.store.history_complete(channel_id, t0)

            rows.append(row)
            meta.append((channel_id, t0, complete))

    raw_predictions = service.model.predict(rows) if rows else []

    return [
        ShiftPrediction(
            channel_id=channel_id,
            t0=t0,
            probability=service._probability_from_raw(float(raw), horizon_hours),
            history_complete=complete,
        )
        for (channel_id, t0, complete), raw in zip(meta, raw_predictions)
    ]
