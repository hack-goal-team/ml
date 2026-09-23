"""Healthcheck контейнера: python -m inference.health, код выхода 0/1.

Смотрит на файлы runner-а, а не на prediction_log: без новых логов
прогнозов нет, и тихий час — это норма, а не сбой.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from inference.runner import HEARTBEAT_FILE, READY_FILE


def check(
    runtime_dir: Path,
    max_age_seconds: float,
    now: float | None = None,
) -> str | None:
    """None — здоров, иначе причина."""
    if not (runtime_dir / READY_FILE).exists():
        return "startup not complete (warmup in progress or failed)"

    try:
        mtime = (runtime_dir / HEARTBEAT_FILE).stat().st_mtime
    except FileNotFoundError:
        return "no heartbeat yet"

    # Такт обновляет heartbeat только после успешного прохода по БД.
    age = (time.time() if now is None else now) - mtime
    if age > max_age_seconds:
        return f"heartbeat is {age:.0f}s old (limit {max_age_seconds:.0f}s)"

    return None


def main() -> int:
    problem = check(
        Path(os.environ.get("INFERENCE_RUNTIME_DIR") or "runtime"),
        float(os.environ.get("HEALTH_MAX_AGE_SECONDS") or 120),
    )
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
