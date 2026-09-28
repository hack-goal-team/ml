"""Эпизод — переход канала из нормы в тревогу; повторные тревоги подряд
эпизод не продлевают и не открывают заново.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from tests.backtest.data import ID_COL


@dataclass(slots=True)
class ChannelTimeline:
    ts: list[datetime]
    is_alarm: list[bool]
    alarm_ts: list[datetime]
    episode_starts: list[datetime]
    episode_values: list[str]
    seed_is_alarm: bool | None


def build_timelines(
    events_df: pl.DataFrame, seed_states: dict[int, bool]
) -> dict[int, ChannelTimeline]:
    timelines: dict[int, ChannelTimeline] = {}

    for (channel_id,), group in events_df.partition_by(
        ID_COL, as_dict=True, maintain_order=True
    ).items():
        timestamps = group["ts"].to_list()
        is_alarm = group["тревожное"].to_list()
        seed = seed_states.get(int(channel_id))

        starts, start_values, in_alarm = [], [], bool(seed)
        for ts, alarm, value in zip(timestamps, is_alarm, group["значение_датчика"]):
            if alarm and not in_alarm:
                starts.append(ts)
                start_values.append(value)
            in_alarm = alarm

        timelines[int(channel_id)] = ChannelTimeline(
            ts=timestamps,
            is_alarm=is_alarm,
            alarm_ts=[ts for ts, alarm in zip(timestamps, is_alarm) if alarm],
            episode_starts=starts,
            episode_values=start_values,
            seed_is_alarm=seed,
        )

    return timelines


def alarms_in_window(timeline: ChannelTimeline, start: datetime, end: datetime) -> int:
    """Число тревожных событий с ts в [start, end)."""
    return bisect.bisect_left(timeline.alarm_ts, end) - bisect.bisect_left(timeline.alarm_ts, start)


def state_at(timeline: ChannelTimeline, t0: datetime) -> bool | None:
    """Тревожное/норма перед t0 (последнее событие с ts < t0, иначе seed)."""
    idx = bisect.bisect_left(timeline.ts, t0) - 1
    return timeline.is_alarm[idx] if idx >= 0 else timeline.seed_is_alarm


def first_episode_after(
    timeline: ChannelTimeline, t0: datetime, horizon: timedelta
) -> datetime | None:
    """Первое начало эпизода в (t0, t0 + horizon], иначе None."""
    lo = bisect.bisect_right(timeline.episode_starts, t0)
    if lo < len(timeline.episode_starts) and timeline.episode_starts[lo] <= t0 + horizon:
        return timeline.episode_starts[lo]
    return None


def episode_value_breakdown(
    timelines: dict[int, ChannelTimeline], start: datetime, end: datetime
) -> pl.DataFrame:
    """Начала эпизодов в [start, end) по значению датчика: часть — «Неисправен»."""
    df = pl.DataFrame(
        [
            {"значение_датчика": value}
            for timeline in timelines.values()
            for ts, value in zip(timeline.episode_starts, timeline.episode_values)
            if start <= ts < end
        ]
    )
    return (
        df.group_by("значение_датчика")
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / df.height).alias("share"))
        .sort("n", descending=True)
    )
