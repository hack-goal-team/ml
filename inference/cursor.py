"""Позиция чтения events, устойчивая к коммитам не по порядку id."""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Iterable


class EventCursor:
    """Какие строки events уже обработаны.

    id выдаётся при INSERT, а видна строка после COMMIT: консьюмер и
    импорт коммитятся не по порядку id, и `id > max` теряет строки.
    Поэтому всё выше low перечитывается, а обработанное отсекает seen.
    """

    def __init__(
        self,
        low: int,
        seen: Iterable[int] = (),
        lag_seconds: float = 300.0,
        max_seen: int = 200_000,
    ) -> None:
        self.low = low
        self.seen = {event_id for event_id in seen if event_id > low}
        self.high = max(self.seen, default=low)
        self.lag_seconds = lag_seconds
        self.max_seen = max_seen
        self.dirty = False
        self._history: deque[tuple[float, int]] = deque()

    def mark(self, event_id: int) -> None:
        if event_id > self.low:
            self.seen.add(event_id)
            self.high = max(self.high, event_id)
            self.dirty = True

    def advance(self, now: float) -> None:
        """Поднимает low до high, который видели не меньше lag секунд назад.

        Все id ниже того high уже были выданы; транзакция, не успевшая
        закоммитить их за lag, теряется — это и есть граница гарантии.
        """
        self._history.append((now, self.high))
        low = self.low

        while self._history and self._history[0][0] <= now - self.lag_seconds:
            low = max(low, self._history.popleft()[1])

        # Предохранитель памяти: при лавине строк держим max_seen старших.
        if len(self.seen) > self.max_seen:
            low = max(low, sorted(self.seen)[-self.max_seen - 1])

        if low > self.low:
            self.low = low
            self.seen = {event_id for event_id in self.seen if event_id > low}
            self.dirty = True

    def save(self, path: Path) -> None:
        # tmp + fsync + rename: после сбоя на диске старая или новая версия.
        tmp = path.with_name(path.name + ".tmp")

        with tmp.open("w", encoding="utf-8") as file:
            json.dump({"low": self.low, "seen": sorted(self.seen)}, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp, path)
        self.dirty = False

    @classmethod
    def load(
        cls,
        path: Path,
        lag_seconds: float = 300.0,
        max_seen: int = 200_000,
    ) -> "EventCursor | None":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                int(data["low"]),
                (int(event_id) for event_id in data["seen"]),
                lag_seconds=lag_seconds,
                max_seen=max_seen,
            )
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Broken cursor file {path}: remove it to start "
                "from current max(events.id)"
            ) from exc
