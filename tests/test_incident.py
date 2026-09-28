from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from inference.config import InferenceSettings
from inference.incident import TargetHistory, incident_class, is_target_alarm
from inference.runner import Runner, RunnerSettings
from tests.test_runner import CHANNEL, db, predictions
from tests.test_service import PG_ENV

ROOT = Path(__file__).resolve().parent.parent


def test_v4_500_artifact_is_pinned() -> None:
    digest = hashlib.sha256(
        (ROOT / "data/incident4/model_500.cbm").read_bytes()
    ).hexdigest()
    assert digest == "178c874ca373a5c48327d4a221379ea9ee166480385393425b96dd5002ce21f9"


def test_incident_mapping_and_prior_target(tmp_path: Path) -> None:
    assert incident_class("КД АВ") == "INTRUSION"
    assert incident_class("Датчик дыма") == "FIRE"
    assert incident_class("Датчик температуры") is None
    assert is_target_alarm("КД АВ", "Не замкнут", True)
    assert not is_target_alarm("КД АВ", "Неисправен", True)
    assert not is_target_alarm("КД АВ", "Не замкнут", False)

    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    history = TargetHistory()
    assert history.hours_before(CHANNEL, now) is None
    history.mark(CHANNEL, now - timedelta(hours=24), 11)
    assert history.hours_before(CHANNEL, now) == 24
    history.mark(CHANNEL, now - timedelta(hours=48), 12)
    assert history.hours_before(CHANNEL, now) == 24
    path = tmp_path / "targets.json"
    history.save(path, 20, {21})
    assert TargetHistory.load(path, 20, {21}) == history
    assert TargetHistory.load(path, 20, set()) is None


def test_incident_runner_routes_and_caps_shap(db, tmp_path: Path) -> None:
    settings = InferenceSettings.from_env({
        **PG_ENV,
        "WEATHER_DISTRICT_ID": "5773",
        "PREDICTION_KIND": "INCIDENT",
        "MODEL_PATH": str(ROOT / "data/incident4/model_500.cbm"),
    })
    settings = replace(
        settings,
        service=replace(settings.service, shap_auto_threshold=0.0),
    )
    runner = Runner(
        settings,
        RunnerSettings(runtime_dir=tmp_path / "runtime", shap_max_per_tick=1),
        connect=db.inference,
    )
    now = datetime.now(timezone.utc)
    with db.admin() as conn:
        conn.execute(
            "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value) "
            "VALUES (10001, %s, %s, true, 'Не замкнут')",
            (CHANNEL, now - timedelta(hours=24)),
        )

    with db.inference() as conn:
        runner.start(conn)
        assert runner.targets.hours_before(CHANNEL, now) == 24
        with db.admin() as admin:
            for offset in range(40):
                event_id = 10002 + offset
                value = "Не замкнут" if offset == 0 else "Норма"
                alarm = offset == 0
                admin.execute(
                    "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (event_id, CHANNEL, now + timedelta(seconds=offset),
                     alarm, value),
                )
        runner.tick(conn)

    rows = predictions(db)
    assert len(rows) == 40
    assert {row[2] for row in rows} == {"INTRUSION"}
    assert all(row[5].startswith("incident4-v4-") for row in rows)
    assert sum(row[6] is not None for row in rows) == 1
    assert runner.last_tick.shap_skipped == 39
    assert runner.targets.hours_before(CHANNEL, now + timedelta(hours=1)) == 1
    runner.service.close()


def test_history_ignores_excluded_2021(db) -> None:
    with db.admin() as conn:
        for event_id, year in [(20001, 2020), (20002, 2021)]:
            conn.execute(
                "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value) "
                "VALUES (%s, %s, %s, true, 'Не замкнут')",
                (event_id, CHANNEL, datetime(year, 6, 1, tzinfo=timezone.utc)),
            )
        low = conn.execute("SELECT max(id) FROM events").fetchone()[0]
    with db.inference() as conn:
        history = TargetHistory.from_db(conn, low, set())
    assert history.latest[CHANNEL][0].year == 2020


def test_history_waits_for_alarm_backfill(db) -> None:
    with db.admin() as conn:
        conn.execute("DELETE FROM alarm_backfill")
    with db.inference() as conn:
        with pytest.raises(RuntimeError, match="alarm_backfill"):
            TargetHistory.from_db(conn, 0, set())
