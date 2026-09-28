from __future__ import annotations

import itertools
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest

from inference.config import InferenceSettings
from inference.runner import Runner, RunnerSettings
from tests.test_service import PG_ENV

CHANNEL = 120578
OTHER = 120579
NEW_CHANNEL = 777
SNAPSHOT = datetime(2026, 9, 1, tzinfo=timezone.utc)

_event_ids = itertools.count(1)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def add_channel(conn: psycopg.Connection, channel_id: int) -> None:
    conn.execute(
        "INSERT INTO dim_channels VALUES "
        "(%s, %s, 'Охранная подсистема', 'КД АВ', '15-11.1.131.2.', 'КД АВ', 20)",
        (channel_id, SNAPSHOT),
    )


def add_event(
    conn: psycopg.Connection,
    channel_id: int,
    ts: datetime,
    value: str = "Норма",
) -> None:
    conn.execute(
        "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value) "
        "VALUES (%s, %s, %s, false, %s)",
        (next(_event_ids), channel_id, ts, value),
    )


def predictions(pg) -> list[tuple]:
    with pg.admin() as conn:
        return conn.execute(
            "SELECT target_kind, target_ref, incident_type, probability, "
            "horizon_until, model_version, shap FROM prediction_log "
            "ORDER BY prediction_id"
        ).fetchall()


@pytest.fixture
def db(pg):
    with pg.admin() as conn:
        conn.execute(
            "INSERT INTO districts VALUES (5773, 'Район', 55.75583, 37.61778)"
        )
        conn.execute(
            "INSERT INTO dim_objects VALUES (20, %s, 3, 5773, 'controlHouse', 'x')",
            (SNAPSHOT,),
        )
        add_channel(conn, CHANNEL)
        add_channel(conn, OTHER)
        conn.execute("INSERT INTO reason_codes VALUES ('FALSE', 'Ложное')")
    return pg


@pytest.fixture
def make_runner(db, tmp_path: Path):
    settings = InferenceSettings.from_env(
        {**PG_ENV, "WEATHER_DISTRICT_ID": "5773"}
    )
    runners: list[Runner] = []

    def make(threshold: float = 0.5, **options) -> Runner:
        service = replace(settings.service, shap_auto_threshold=threshold)
        runner = Runner(
            replace(settings, service=service),
            RunnerSettings(runtime_dir=tmp_path / "runtime", **options),
            connect=db.inference,
            monotonic=FakeClock(),
        )
        runners.append(runner)
        return runner

    yield make

    for runner in runners:
        if runner.service is not None:
            runner.service.close()


def started(make_runner, db, **kwargs) -> tuple[Runner, psycopg.Connection]:
    runner = make_runner(**kwargs)
    conn = db.inference()
    runner.start(conn)
    return runner, conn


def test_tick_writes_prediction_rows(db, make_runner) -> None:
    runner, conn = started(make_runner, db, threshold=0.0)
    ts = datetime.now(timezone.utc) - timedelta(minutes=5)

    with db.admin() as admin:
        add_event(admin, CHANNEL, ts, "Неисправен")
        add_event(admin, CHANNEL, ts + timedelta(seconds=1), "12.5")

    runner.tick(conn)
    rows = predictions(db)

    assert len(rows) == 2
    kind, ref, incident, probability, until, version, shap = rows[0]
    assert (kind, ref, incident) == ("channel", str(CHANNEL), "CHANNEL_EVENT")
    assert Decimal(0) <= probability <= Decimal(1)
    assert probability == probability.quantize(Decimal("0.0001"))
    assert until == ts + timedelta(hours=30)
    assert version.startswith("catboost-aft-")

    # Порог 0 — SHAP на каждом прогнозе, формат из 014.
    assert set(shap) == {"base_value", "top"}
    assert 0 < len(shap["top"]) <= 10
    assert set(shap["top"][0]) == {
        "feature", "feature_value", "shap_raw", "risk_direction",
        "probability_delta",
    }
    raw = [abs(item["shap_raw"]) for item in shap["top"]]
    assert raw == sorted(raw, reverse=True)
    assert "null" not in [item["feature_value"] for item in shap["top"]]


def test_equipment_reads_events_before_incident_migration(db, make_runner) -> None:
    with db.admin() as admin:
        admin.execute("ALTER TABLE events DROP COLUMN journal_is_alarm")
    runner, conn = started(make_runner, db)
    with db.admin() as admin:
        add_event(admin, CHANNEL, datetime.now(timezone.utc), "Норма")
    runner.tick(conn)
    assert len(predictions(db)) == 1


def test_shap_null_below_threshold(db, make_runner) -> None:
    runner, conn = started(make_runner, db, threshold=1.0)

    with db.admin() as admin:
        add_event(admin, CHANNEL, datetime.now(timezone.utc))

    runner.tick(conn)
    assert [row[6] for row in predictions(db)] == [None]


def test_out_of_order_commit_is_not_lost(db, make_runner) -> None:
    runner, conn = started(make_runner, db)
    ts = datetime.now(timezone.utc) - timedelta(minutes=1)

    # A получает id раньше, но коммитит позже B.
    slow = db.admin()
    slow.autocommit = False
    add_event(slow, CHANNEL, ts)
    with db.admin() as fast:
        add_event(fast, OTHER, ts)

    runner.tick(conn)
    assert [row[1] for row in predictions(db)] == [str(OTHER)]

    slow.commit()
    slow.close()
    runner.monotonic.now += 60
    runner.tick(conn)
    runner.tick(conn)

    assert sorted(row[1] for row in predictions(db)) == [
        str(CHANNEL), str(OTHER),
    ]


def test_stale_event_updates_state_only(db, make_runner) -> None:
    runner, conn = started(make_runner, db)
    now = datetime.now(timezone.utc)

    with db.admin() as admin:
        add_event(admin, CHANNEL, now - timedelta(hours=100))
        add_event(admin, CHANNEL, now - timedelta(hours=40))

    runner.tick(conn)

    assert predictions(db) == []
    # Строка старше окна не читается, 40 ч назад — только в окна.
    assert len(runner.service.store.states[CHANNEL].events) == 1


def test_out_of_order_event_is_skipped(db, make_runner) -> None:
    runner, conn = started(make_runner, db)
    now = datetime.now(timezone.utc)

    with db.admin() as admin:
        add_event(admin, CHANNEL, now)
    runner.tick(conn)

    with db.admin() as admin:
        add_event(admin, CHANNEL, now - timedelta(minutes=10))
    runner.tick(conn)

    assert len(predictions(db)) == 1


def test_unknown_channel_picked_up_after_reload(db, make_runner) -> None:
    runner, conn = started(make_runner, db, metadata_retry_seconds=600)
    now = datetime.now(timezone.utc)

    with db.admin() as admin:
        add_event(admin, NEW_CHANNEL, now - timedelta(minutes=3))
    runner.monotonic.now += 700
    runner.tick(conn)

    # Перечитали реестр, канала там нет — пропуск; повтор ограничен.
    with db.admin() as admin:
        add_channel(admin, NEW_CHANNEL)
        add_event(admin, NEW_CHANNEL, now - timedelta(minutes=2))
    runner.monotonic.now += 100
    runner.tick(conn)
    assert predictions(db) == []
    assert NEW_CHANNEL not in runner.service.store.states

    with db.admin() as admin:
        add_event(admin, NEW_CHANNEL, now - timedelta(minutes=1))
    runner.monotonic.now += 600
    runner.tick(conn)

    assert [row[1] for row in predictions(db)] == [str(NEW_CHANNEL)]


def test_ttl_keeps_decisions_and_foreign_rows(db, make_runner) -> None:
    runner, conn = started(make_runner, db)
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    future = past + timedelta(hours=1)
    version = runner.settings.model_version

    with db.admin() as admin:
        for ref, until, model in [
            ("expired", past, version),
            ("decided", past, version),
            ("old-model", past, "catboost-aft-000000000000"),
            ("mock", past, "mock-0"),
            ("alive", future, version),
        ]:
            admin.execute(
                "INSERT INTO prediction_log (target_kind, target_ref, "
                "incident_type, probability, horizon_until, model_version) "
                "VALUES ('channel', %s, 'CHANNEL_EVENT', 0.5, %s, %s)",
                (ref, until, model),
            )
        admin.execute(
            "INSERT INTO decisions_on_prediction (prediction_id, "
            "reason_code, decided_by) SELECT prediction_id, 'FALSE', 'op' "
            "FROM prediction_log WHERE target_ref = 'decided'"
        )

    runner.tick(conn)

    assert sorted(row[1] for row in predictions(db)) == [
        "alive", "decided", "mock",
    ]


def test_restart_resumes_without_duplicates(db, make_runner) -> None:
    now = datetime.now(timezone.utc)

    with db.admin() as admin:
        add_event(admin, CHANNEL, now - timedelta(hours=2))

    # Первый запуск: бэклог только в прогрев, без прогнозов.
    first, conn = started(make_runner, db)
    first.tick(conn)
    assert predictions(db) == []
    assert len(first.service.store.states[CHANNEL].events) == 1

    with db.admin() as admin:
        add_event(admin, CHANNEL, now - timedelta(minutes=2))
    first.tick(conn)
    assert len(predictions(db)) == 1
    first.service.close()
    first.service = None

    with db.admin() as admin:
        add_event(admin, CHANNEL, now - timedelta(minutes=1))

    second, conn = started(make_runner, db)
    # Прогрев взял оба учтённых лога, но не новый — его прогнозирует цикл.
    assert len(second.service.store.states[CHANNEL].events) == 2
    second.tick(conn)
    second.tick(conn)

    assert len(predictions(db)) == 2
    assert len(second.service.store.states[CHANNEL].events) == 3


def test_run_reconnects_after_db_error(db, make_runner) -> None:
    calls = []

    def flaky() -> psycopg.Connection:
        calls.append(1)
        if len(calls) == 1:
            raise psycopg.OperationalError("connection refused")
        return db.inference()

    runner = make_runner(poll_seconds=0.01)
    runner.connect = flaky
    heartbeat = runner.path("heartbeat")

    thread = threading.Thread(target=runner.run)
    thread.start()
    try:
        for _ in range(600):
            if heartbeat.exists():
                break
            threading.Event().wait(0.05)
    finally:
        runner.stop_event.set()
        thread.join(timeout=30)

    assert heartbeat.exists()
    assert (runner.options.runtime_dir / "ready").exists()
    assert len(calls) >= 2


def test_cursor_ahead_of_database_is_reset(db, make_runner) -> None:
    runner = make_runner()
    runner.options.runtime_dir.mkdir(parents=True)
    runner.path("events_cursor.json").write_text(
        '{"low": 1000000, "seen": []}', encoding="utf-8"
    )
    conn = db.inference()
    runner.start(conn)

    with db.admin() as admin:
        add_event(admin, CHANNEL, datetime.now(timezone.utc))
    runner.tick(conn)

    assert len(predictions(db)) == 1


def test_broken_shap_is_counted_not_skipped(db, make_runner, monkeypatch) -> None:
    runner, conn = started(make_runner, db, threshold=0.0)

    def broken(shap, cat_features):
        raise KeyError("probability_delta_from_component")

    monkeypatch.setattr("inference.runner.shap_contract", broken)
    with db.admin() as admin:
        add_event(admin, CHANNEL, datetime.now(timezone.utc))
    runner.tick(conn)

    # Сбой сборки строки — не «неизвестный канал» и не 409.
    tick = runner.last_tick
    assert (tick.failed_write_prep, tick.skipped_unknown, tick.predicted) == (
        1, 0, 0,
    )
    assert predictions(db) == []


def test_run_clears_stale_health_files(db, make_runner) -> None:
    runner = make_runner()
    runner.options.runtime_dir.mkdir(parents=True)
    for name in ("ready", "heartbeat"):
        runner.path(name).touch()

    # БД недоступна: старые файлы volume не должны делать health зелёным.
    def down() -> psycopg.Connection:
        runner.stop_event.set()
        raise psycopg.OperationalError("connection refused")

    runner.connect = down
    runner.run()

    assert not runner.path("ready").exists()
    assert not runner.path("heartbeat").exists()
