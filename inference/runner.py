"""Цикл inference: events → PredictionService → prediction_log.

Запуск: python -m inference.runner. Источник правды — Postgres:
rolling-state прогревается из events, pickle ядра не читается.
"""
from __future__ import annotations

import logging
import os
import signal
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import psycopg

from app import DATE_COL, HORIZON_HOURS_FIELD, ID_COL, TIME_COL, VALUE_COL
from app import PredictionService
from inference.config import InferenceSettings
from inference.config import connect as pg_connect
from inference.cursor import EventCursor
from inference.health import HEARTBEAT_FILE, READY_FILE
from inference.predictions import (
    PredictionRow,
    delete_expired,
    shap_contract,
    to_probability,
    write_predictions,
)
from inference.reference import fetch_metadata
from inference.service import build_service
from inference.timeutil import to_model_fields, to_model_time

log = logging.getLogger("inference")

CURSOR_FILE = "events_cursor.json"

SELECT_MAX_ID = "SELECT coalesce(max(id), 0) FROM events"

# Прогрев — только то, что уже учтено курсором: остальное придёт
# в цикле, и второй update задвоил бы окна.
SELECT_WARMUP = """
    SELECT channel_id, ts, raw_value FROM events
    WHERE ts >= %(since)s
      AND (id <= %(low)s OR id = ANY(%(seen)s::bigint[]))
    ORDER BY ts, id
"""

# Фильтр по ts отсекает старые партиции и импорт истории вне окон.
SELECT_NEW = """
    SELECT id, channel_id, ts, raw_value FROM events
    WHERE ts >= %(since)s AND id > %(low)s
      AND id <> ALL(%(seen)s::bigint[])
    ORDER BY id
    LIMIT %(limit)s
"""


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    return float(env.get(name) or default)


@dataclass(slots=True, frozen=True)
class RunnerSettings:
    runtime_dir: Path = Path("runtime")
    poll_seconds: float = 2.0
    batch_size: int = 5000
    reorder_lag_seconds: float = 300.0
    ttl_interval_seconds: float = 60.0
    metadata_refresh_seconds: float = 3600.0
    metadata_retry_seconds: float = 600.0
    backoff_max_seconds: float = 60.0

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] = os.environ,
    ) -> "RunnerSettings":
        return cls(
            runtime_dir=Path(env.get("INFERENCE_RUNTIME_DIR") or "runtime"),
            poll_seconds=_env_float(env, "POLL_INTERVAL_SECONDS", 2.0),
            batch_size=int(env.get("EVENTS_BATCH_SIZE") or 5000),
            reorder_lag_seconds=_env_float(
                env, "EVENTS_REORDER_LAG_SECONDS", 300.0
            ),
            ttl_interval_seconds=_env_float(env, "TTL_INTERVAL_SECONDS", 60.0),
            metadata_refresh_seconds=_env_float(
                env, "METADATA_REFRESH_SECONDS", 3600.0
            ),
            metadata_retry_seconds=_env_float(
                env, "METADATA_RETRY_SECONDS", 600.0
            ),
        )


@dataclass(slots=True)
class TickCounters:
    read: int = 0
    predicted: int = 0
    skipped_stale: int = 0
    skipped_ooo: int = 0
    skipped_unknown: int = 0
    failed: int = 0
    failed_write_prep: int = 0
    shap: int = 0
    expired: int = 0
    write_ms: float = 0.0
    lag_seconds: float | None = None
    ticks: int = 0

    def add(self, other: "TickCounters") -> None:
        # Сумма за окно лога; lag — последнего такта с данными.
        for name in self.__dataclass_fields__:
            if name != "lag_seconds":
                setattr(self, name, getattr(self, name) + getattr(other, name))
        if other.lag_seconds is not None:
            self.lag_seconds = other.lag_seconds


@dataclass(slots=True, frozen=True)
class EventRow:
    id: int
    channel_id: int
    ts: datetime
    raw_value: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Runner:
    def __init__(
        self,
        settings: InferenceSettings,
        options: RunnerSettings,
        connect: Callable[[], psycopg.Connection] = pg_connect,
        clock: Callable[[], datetime] = utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        # Pickle ядра не нужен и опасен: устаревший state наложился бы
        # на прогрев. Путь в пустом каталоге не существует никогда.
        no_state = Path(tempfile.mkdtemp(prefix="inference-")) / "state.pkl"
        service = replace(
            settings.service,
            state_path=no_state,
            state_save_every_n=0,
            log_path=options.runtime_dir / "service.log",
        )
        self.settings = replace(settings, service=service)
        self.options = options
        self.connect = connect
        self.clock = clock
        self.monotonic = monotonic
        self.stop_event = threading.Event()
        self.service: PredictionService | None = None
        self.cursor: EventCursor | None = None
        self.pending: list[PredictionRow] = []
        self.horizon = timedelta(hours=self.settings.horizon_hours)
        self.window = timedelta(hours=72)
        self._metadata_at = 0.0
        self._ttl_at = float("-inf")
        self.totals = TickCounters()
        self.last_tick = TickCounters()
        self._logged_at = float("-inf")
        self._traceback_at = float("-inf")

    def path(self, name: str) -> Path:
        return self.options.runtime_dir / name

    # --- старт ---------------------------------------------------------

    def start(self, conn: psycopg.Connection) -> None:
        self.options.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.path(READY_FILE).unlink(missing_ok=True)

        service = build_service(self.settings, connect=self.connect)
        try:
            # Окно прогрева — самое длинное rolling-окно модели (3 дня).
            seconds = service.store.max_window_seconds
            self.window = timedelta(seconds=seconds) if seconds else self.window
            cursor = EventCursor.load(
                self.path(CURSOR_FILE),
                lag_seconds=self.options.reorder_lag_seconds,
            )
            max_id = conn.execute(SELECT_MAX_ID).fetchone()[0]
            if cursor is not None and cursor.high > max_id:
                # База пересоздана, id пошли заново: старый курсор
                # молча пропускал бы все новые строки.
                log.warning("cursor high=%d > max(id)=%d, reset", cursor.high,
                            max_id)
                cursor = None
            fresh = cursor is None
            if cursor is None:
                # Первый запуск: бэклог не прогнозируем, только прогрев.
                low = max_id
                cursor = EventCursor(
                    low, lag_seconds=self.options.reorder_lag_seconds
                )
            rows = self._warmup(conn, service, cursor)
        except BaseException:
            service.close()
            raise

        # Лог ядра на каждый прогноз — 170 тыс. строк в сутки; счётчики
        # есть в строке тика, ядру оставляем предупреждения.
        service.logger.setLevel(logging.WARNING)
        self.service = service
        self.cursor = cursor
        self._metadata_at = self.monotonic()
        cursor.save(self.path(CURSOR_FILE))
        self.path(READY_FILE).touch()
        log.info(
            "started fresh=%s low=%d seen=%d warmup_rows=%d channels=%d "
            "model_version=%s window_h=%.0f",
            fresh, cursor.low, len(cursor.seen), rows,
            len(service.metadata), self.settings.model_version,
            self.window.total_seconds() / 3600,
        )

    def _warmup(
        self,
        conn: psycopg.Connection,
        service: PredictionService,
        cursor: EventCursor,
    ) -> int:
        params = {
            "since": self.clock() - self.window,
            "low": cursor.low,
            "seen": list(cursor.seen),
        }

        def payloads() -> Iterator[dict[str, Any]]:
            with conn.transaction():
                # Сортировка трёх дней может не влезть в 30 s сессии.
                conn.execute("SET LOCAL statement_timeout = '10min'")
                with conn.cursor(name="warmup") as cur:
                    cur.itersize = 10_000
                    cur.execute(SELECT_WARMUP, params)
                    for channel_id, ts, raw_value in cur:
                        if self.stop_event.is_set():
                            raise InterruptedError("stopped during warmup")
                        date_value, time_value = to_model_fields(ts)
                        yield {
                            ID_COL: channel_id,
                            DATE_COL: date_value,
                            TIME_COL: time_value,
                            VALUE_COL: raw_value,
                        }

        return service.warmup(payloads())

    # --- такт ----------------------------------------------------------

    def tick(self, conn: psycopg.Connection) -> bool:
        """Один проход; True — страница полная, читать дальше без паузы."""
        assert self.service is not None and self.cursor is not None
        counters = TickCounters(ticks=1)

        # Прогнозы прошлого такта, не записанные из-за сбоя БД.
        counters.write_ms += self._flush(conn)

        if (
            self.monotonic() - self._metadata_at
            >= self.options.metadata_refresh_seconds
        ):
            self._reload_metadata(conn, "periodic")

        if self.monotonic() - self._ttl_at >= self.options.ttl_interval_seconds:
            counters.expired = delete_expired(conn, self.settings.model_version)
            self._ttl_at = self.monotonic()

        events = self._fetch(conn)
        counters.read = len(events)
        now = self.clock()

        for event in sorted(events, key=lambda row: (row.ts, row.id)):
            self._process(conn, event, now, counters)
            self.cursor.mark(event.id)

        if events:
            newest = max(event.ts for event in events)
            counters.lag_seconds = round((now - newest).total_seconds(), 1)

        counters.write_ms += self._flush(conn)
        self.cursor.advance(self.monotonic())
        if self.cursor.dirty:
            self.cursor.save(self.path(CURSOR_FILE))

        self.last_tick = counters
        self._log_tick(counters)
        return len(events) >= self.options.batch_size

    def _fetch(self, conn: psycopg.Connection) -> list[EventRow]:
        assert self.cursor is not None
        rows = conn.execute(
            SELECT_NEW,
            {
                "since": self.clock() - self.window,
                "low": self.cursor.low,
                "seen": list(self.cursor.seen),
                "limit": self.options.batch_size,
            },
        ).fetchall()
        return [EventRow(*row) for row in rows]

    def _process(
        self,
        conn: psycopg.Connection,
        event: EventRow,
        now: datetime,
        counters: TickCounters,
    ) -> None:
        service = self.service
        assert service is not None

        if event.channel_id not in service.metadata:
            # Реестры обновляются раз в месяц: новый канал — повод
            # перечитать их, но не чаще раза в metadata_retry_seconds.
            if (
                self.monotonic() - self._metadata_at
                >= self.options.metadata_retry_seconds
            ):
                self._reload_metadata(conn, "unknown_channel")
            if event.channel_id not in service.metadata:
                counters.skipped_unknown += 1
                return

        horizon_until = event.ts + self.horizon

        if horizon_until < now:
            # Прогноз родился бы истёкшим; окна всё равно обновляем.
            counters.skipped_stale += 1
            with service.lock:
                try:
                    service.store.update(
                        event.channel_id,
                        to_model_time(event.ts),
                        event.raw_value,
                    )
                except ValueError:
                    pass
            return

        date_value, time_value = to_model_fields(event.ts)
        try:
            result = service.predict(
                {
                    ID_COL: event.channel_id,
                    DATE_COL: date_value,
                    TIME_COL: time_value,
                    VALUE_COL: event.raw_value,
                    HORIZON_HOURS_FIELD: self.settings.horizon_hours,
                    "add_shap": "auto",
                }
            )
        except KeyError:
            counters.skipped_unknown += 1
            return
        except ValueError:
            # Лог старше уже учтённого по каналу (в API это 409).
            counters.skipped_ooo += 1
            return
        except Exception:  # noqa: BLE001
            # Битая строка не должна крутить рестарты: считаем и идём дальше.
            self._log_failure("predict", event.id)
            counters.failed += 1
            return

        # Отдельно от predict: сломанный формат ответа ядра (NaN, новое
        # имя ключа shap) — наш сбой, а не пропуск unknown/ooo.
        try:
            shap = result.get("shap")
            row = PredictionRow(
                channel_id=event.channel_id,
                probability=to_probability(result["probability"]),
                horizon_until=horizon_until,
                shap=None if shap is None else shap_contract(
                    shap, self._cat_features(service)
                ),
            )
        except Exception:  # noqa: BLE001
            self._log_failure("write_prep", event.id)
            counters.failed_write_prep += 1
            return

        self.pending.append(row)
        counters.predicted += 1
        counters.shap += row.shap is not None

    def _log_failure(self, stage: str, event_id: int) -> None:
        # Traceback раз в минуту: при системной поломке он был бы на
        # каждом логе. Остальные случаи видны счётчиками в строке tick.
        now = self.monotonic()
        if now - self._traceback_at >= 60:
            self._traceback_at = now
            log.exception("%s failed event_id=%d", stage, event_id)

    @staticmethod
    def _cat_features(service: PredictionService) -> frozenset[str]:
        return frozenset(
            service.feature_names[idx] for idx in service.cat_feature_indices
        )

    def _flush(self, conn: psycopg.Connection) -> float:
        if not self.pending:
            return 0.0

        started = time.perf_counter()
        write_predictions(conn, self.pending, self.settings.model_version)
        self.pending.clear()
        return round((time.perf_counter() - started) * 1000, 1)

    def _reload_metadata(self, conn: psycopg.Connection, reason: str) -> None:
        service = self.service
        assert service is not None
        metadata = fetch_metadata(conn, service.compiled.static_features)

        # Rolling-state не трогаем: меняется только справочная часть.
        with service.lock:
            service.metadata = metadata
            service.store.metadata = metadata

        self._metadata_at = self.monotonic()
        log.info("metadata_reloaded reason=%s channels=%d", reason, len(metadata))

    def _log_tick(self, counters: TickCounters) -> None:
        # Одна строка в минуту с суммой за окно: при 2 лог/с почти каждый
        # такт с данными, и построчный лог дал бы 43 тыс. строк в сутки.
        self.totals.add(counters)
        now = self.monotonic()
        if now - self._logged_at < 60:
            return

        self._logged_at = now
        data = asdict(self.totals)
        data["write_ms"] = round(data["write_ms"], 1)
        assert self.cursor is not None
        data.update(low=self.cursor.low, seen=len(self.cursor.seen))
        log.info("tick %s", " ".join(f"{k}={v}" for k, v in data.items()))
        self.totals = TickCounters()

    # --- жизненный цикл ------------------------------------------------

    def heartbeat(self) -> None:
        self.path(HEARTBEAT_FILE).touch()

    def run(self) -> None:
        conn: psycopg.Connection | None = None
        backoff = 1.0

        # runtime/ — volume: файлы прошлого контейнера сделали бы health
        # зелёным, пока этот ещё не подключился к БД.
        self.options.runtime_dir.mkdir(parents=True, exist_ok=True)
        for name in (READY_FILE, HEARTBEAT_FILE):
            self.path(name).unlink(missing_ok=True)

        while not self.stop_event.is_set():
            try:
                if conn is None or conn.closed:
                    conn = self.connect()
                if self.service is None:
                    self.start(conn)
                more = self.tick(conn)
                self.heartbeat()
                backoff = 1.0
            except psycopg.Error as exc:
                # Обрыв БД: переподключение с backoff, а не рестарт контейнера.
                log.warning("db error, retry in %.0fs: %s", backoff, exc)
                if conn is not None:
                    conn.close()
                conn = None
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, self.options.backoff_max_seconds)
                continue
            except InterruptedError:
                break

            if not more:
                self.stop_event.wait(self.options.poll_seconds)

        self._shutdown(conn)

    def _shutdown(self, conn: psycopg.Connection | None) -> None:
        try:
            if conn is not None and not conn.closed and self.cursor is not None:
                self._flush(conn)
                self.cursor.save(self.path(CURSOR_FILE))
        except psycopg.Error as exc:
            # Не записанные строки пересчитаются после рестарта.
            log.warning("final flush failed: %s", exc)
        finally:
            if conn is not None:
                conn.close()
            if self.service is not None:
                self.service.close()
        log.info("stopped")


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main() -> None:
    configure_logging()
    runner = Runner(InferenceSettings.from_env(), RunnerSettings.from_env())

    def stop(signum: int, _frame: Any) -> None:
        log.info("signal %d, stopping", signum)
        runner.stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runner.run()


if __name__ == "__main__":
    main()
