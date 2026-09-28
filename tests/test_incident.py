from __future__ import annotations

import hashlib
import json
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
    seed = hashlib.sha256(
        (ROOT / "data/incident4/target_seed.json").read_bytes()
    ).hexdigest()
    assert seed == "3161acd2ab38454f009aeb2b088ba53d7c673603ea9306aad7d5b4c87ae05deb"


def test_incident_mapping_and_prior_target(tmp_path: Path) -> None:
    assert incident_class("КД АВ") == "INTRUSION"
    assert incident_class("Датчик дыма") == "FIRE"
    assert incident_class("Датчик температуры") is None
    assert is_target_alarm("КД АВ", "Не замкнут", True, True)
    assert not is_target_alarm("КД АВ", "Неисправен", True, True)
    assert not is_target_alarm("КД АВ", "Не замкнут", True, False)
    assert not is_target_alarm("КД АВ", "Норма", False)
    with pytest.raises(RuntimeError, match="Original intrusion"):
        is_target_alarm("КД АВ", "Не замкнут", True)

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
            "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value, journal_is_alarm) "
            "VALUES (10001, %s, %s, true, 'Не замкнут', true)",
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
                    "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value, journal_is_alarm) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (event_id, CHANNEL, now + timedelta(seconds=offset),
                     alarm, value, alarm),
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


def test_seed_excludes_2021() -> None:
    seed = json.loads((ROOT / "data/incident4/target_seed.json").read_text())
    assert seed["cutoff"] == "2026-07-01T00:00:00+03:00"
    assert len(seed["latest"]) > 8000
    assert all(datetime.fromisoformat(row[0]).year != 2021
               for row in seed["latest"].values())


def test_history_waits_for_alarm_backfill(db) -> None:
    with db.admin() as conn:
        conn.execute("DELETE FROM alarm_backfill")
    with db.inference() as conn:
        with pytest.raises(RuntimeError, match="alarm_backfill"):
            TargetHistory.from_db(conn, 0, set())


def test_history_rejects_unrestored_intrusion_flag(db) -> None:
    with db.admin() as conn:
        conn.execute(
            "INSERT INTO events (event_id, channel_id, ts, is_alarm, raw_value) "
            "VALUES (30001, %s, %s, true, 'Обнаружено движение')",
            (CHANNEL, datetime(2026, 9, 1, tzinfo=timezone.utc)),
        )
        low = conn.execute("SELECT max(id) FROM events").fetchone()[0]
    with db.inference() as conn:
        with pytest.raises(RuntimeError, match="Original intrusion"):
            TargetHistory.from_db(conn, low, set())


def test_history_uses_compact_checkpoint(db) -> None:
    at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with db.admin() as conn:
        conn.execute(
            "UPDATE incident_history_checkpoint SET covered_until = %s, "
            "targets = jsonb_build_object(%s::text, jsonb_build_array(%s::timestamptz, 12345)) "
            "WHERE id = 1", (at + timedelta(days=1), str(CHANNEL), at),
        )
        low = conn.execute("SELECT coalesce(max(id), 0) FROM events").fetchone()[0]
    with db.inference() as conn:
        history = TargetHistory.from_db(conn, low, set())
    assert history.hours_before(CHANNEL, at + timedelta(days=2)) == 48
