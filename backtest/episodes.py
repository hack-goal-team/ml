"""Эпизоды тревог и «вчерашние» окна для бейзлайнов и исхода прогноза.

Эпизод — переход канала из нормы в тревогу (тревожное 0 -> 1); повторные
тревожные события подряд эпизод не продлевают и не открывают заново.
Состояние канала перед первым событием окна берётся из
backtest.warmstate.last_state_before — без этого первый тревожный сигнал
в окне ошибочно читался бы как новый эпизод (см. warmstate.py).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime

import polars as pl

ID_COL = "ид_канала_данных"


@dataclass(slots=True)
class ChannelTimeline:
    ts: list[datetime]
    is_alarm: list[bool]
    alarm_ts: list[datetime]
    episode_starts: list[datetime]
    episode_values: list[str]
    seed_is_alarm: bool | None


def _episode_starts(
    timestamps: list[datetime],
    is_alarm: list[bool],
    values: list[str],
    seed: bool | None,
) -> tuple[list[datetime], list[str]]:
    starts: list[datetime] = []
    start_values: list[str] = []
    in_alarm = bool(seed)

    for ts, alarm, value in zip(timestamps, is_alarm, values):
        if alarm and not in_alarm:
            starts.append(ts)
            start_values.append(value)
        in_alarm = alarm

    return starts, start_values


def build_timelines(
    events_df: pl.DataFrame,
    seed_states: dict[int, tuple[bool, str]] | None = None,
) -> dict[int, ChannelTimeline]:
    """events_df уже отсортирован по [ID_COL, ts] (см. backtest.data.load_events)."""
    seed_states = seed_states or {}
    timelines: dict[int, ChannelTimeline] = {}

    for (channel_id,), group in events_df.partition_by(
        ID_COL,
        as_dict=True,
        maintain_order=True,
    ).items():
        timestamps = group["ts"].to_list()
        is_alarm = group["тревожное"].to_list()
        values = group["значение_датчика"].to_list()
        seed = seed_states.get(int(channel_id), (None, None))[0]

        starts, start_values = _episode_starts(timestamps, is_alarm, values, seed)

        timelines[int(channel_id)] = ChannelTimeline(
            ts=timestamps,
            is_alarm=is_alarm,
            alarm_ts=[ts for ts, alarm in zip(timestamps, is_alarm) if alarm],
            episode_starts=starts,
            episode_values=start_values,
            seed_is_alarm=seed,
        )

    return timelines


def alarms_in_window(
    timeline: ChannelTimeline,
    start: datetime,
    end: datetime,
) -> int:
    """Число тревожных событий с ts в [start, end)."""
    lo = bisect.bisect_left(timeline.alarm_ts, start)
    hi = bisect.bisect_left(timeline.alarm_ts, end)
    return hi - lo


def state_at(timeline: ChannelTimeline, t0: datetime) -> bool | None:
    """Тревожное/норма непосредственно перед t0 (событие с ts < t0, либо seed)."""
    idx = bisect.bisect_left(timeline.ts, t0) - 1
    return timeline.is_alarm[idx] if idx >= 0 else timeline.seed_is_alarm


def first_episode_after(
    timeline: ChannelTimeline,
    t0: datetime,
    horizon,
) -> datetime | None:
    """Первое начало нового эпизода в (t0, t0 + horizon], иначе None."""
    lo = bisect.bisect_right(timeline.episode_starts, t0)

    if lo >= len(timeline.episode_starts):
        return None

    candidate = timeline.episode_starts[lo]
    return candidate if candidate <= t0 + horizon else None


def episode_value_breakdown(
    timelines: dict[int, ChannelTimeline],
    window_start: datetime,
    window_end: datetime,
) -> pl.DataFrame:
    """Разбивка начал эпизодов в [window_start, window_end) по значению
    датчика, вызвавшему тревогу — не все эпизоды "тревога" в бытовом
    смысле, часть — «Неисправен» и другие сервисные состояния.
    """
    rows = [
        {"value": value, "ts": ts}
        for timeline in timelines.values()
        for ts, value in zip(timeline.episode_starts, timeline.episode_values)
        if window_start <= ts < window_end
    ]

    if not rows:
        return pl.DataFrame({"значение_датчика": [], "n": [], "share": []})

    df = pl.DataFrame(rows)
    total = df.height

    return (
        df.group_by("value")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / total).alias("share"))
        .sort("n", descending=True)
        .rename({"value": "значение_датчика"})
    )
