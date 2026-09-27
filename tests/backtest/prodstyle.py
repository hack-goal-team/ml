"""«Что стенд реально показывал в момент t0».

app.PredictionService.predict строится на КАЖДОМ событии, не на границе
смены (inference/runner.py:333-360): прогноз пишется в prediction_log с
horizon_until = ts + 30ч (HORIZON_HOURS=30, inference/config.py) и
считается устаревшим после этого срока (predictions.delete_expired). Этот
модуль воспроизводит именно это: для каждой смены t0 берёт последний
прогноз ПО СОБЫТИЮ до t0 и его "возраст" на момент t0.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from app import PredictionService

ID_COL = "ид_канала_данных"
PROD_HORIZON_HOURS = 30.0


@dataclass(slots=True)
class ProdPrediction:
    channel_id: int
    t0: datetime
    probability: float | None
    age_hours: float | None

    @property
    def valid(self) -> bool:
        # Стенд удаляет просроченные прогнозы (delete_expired): диспетчер
        # не видит алерта старше горизонта в 30ч на момент t0.
        return self.age_hours is not None and self.age_hours < PROD_HORIZON_HOURS


def predict_last_event(
    service: PredictionService,
    events_df: pl.DataFrame,
    channel_ids: list[int],
    shift_times: list[datetime],
) -> list[ProdPrediction]:
    parts = events_df.partition_by(ID_COL, as_dict=True, maintain_order=True)

    results: list[ProdPrediction] = []
    rows: list[list] = []
    row_meta: list[tuple[int, datetime, list[datetime]]] = []

    for channel_id in channel_ids:
        group = parts.get((channel_id,))
        ts = group["ts"].to_list() if group is not None else []
        values = group["значение_датчика"].to_list() if group is not None else []

        # Для каждой смены — индекс последнего события со ts < t0.
        due: dict[int, list[datetime]] = {}
        for t0 in shift_times:
            idx = bisect.bisect_left(ts, t0) - 1
            due.setdefault(idx, []).append(t0)

        for t0 in due.get(-1, []):
            results.append(ProdPrediction(channel_id, t0, None, None))

        for idx, (event_ts, value) in enumerate(zip(ts, values)):
            service.store.update(channel_id, event_ts, value)
            t0s = due.get(idx)

            if not t0s:
                continue

            # Событие с тем же ts, что следующее: это ещё не последнее
            # событие тика — переносим смены на следующий индекс.
            if idx + 1 < len(ts) and ts[idx + 1] == event_ts:
                due.setdefault(idx + 1, []).extend(t0s)
                continue

            rows.append(service._build_feature_row(channel_id, event_ts))
            row_meta.append((channel_id, event_ts, t0s))

    raw_predictions = service.model.predict(rows) if rows else []

    for (channel_id, event_ts, t0s), raw in zip(row_meta, raw_predictions):
        probability = service._probability_from_raw(float(raw), PROD_HORIZON_HOURS)

        for t0 in t0s:
            age = (t0 - event_ts).total_seconds() / 3600
            results.append(ProdPrediction(channel_id, t0, probability, age))

    return results
