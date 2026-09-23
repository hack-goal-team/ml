from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from inference.cursor import EventCursor
from inference.health import check
from inference.predictions import shap_contract, to_probability


def test_cursor_keeps_rescan_window_for_lag() -> None:
    cursor = EventCursor(low=100, lag_seconds=300)
    cursor.mark(102)
    cursor.advance(now=0)

    # 101 ещё может закоммититься: low стоит, 102 отсекает seen.
    assert (cursor.low, cursor.seen) == (100, {102})

    cursor.mark(101)
    cursor.mark(105)
    cursor.advance(now=299)
    assert cursor.low == 100

    # Через lag после наблюдения high=102 всё до 102 считается финальным.
    cursor.advance(now=300)
    assert (cursor.low, cursor.seen) == (102, {105})

    cursor.advance(now=600)
    assert (cursor.low, cursor.seen) == (105, set())


def test_cursor_ignores_ids_below_low_and_caps_seen() -> None:
    cursor = EventCursor(low=10, lag_seconds=300, max_seen=3)
    cursor.mark(5)
    assert not cursor.dirty

    for event_id in (11, 12, 13, 14, 15):
        cursor.mark(event_id)
    cursor.advance(now=0)

    assert (cursor.low, cursor.seen) == (12, {13, 14, 15})


def test_cursor_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cursor.json"
    assert EventCursor.load(path) is None

    cursor = EventCursor(low=7, seen=[3, 9, 8])
    cursor.save(path)
    loaded = EventCursor.load(path)

    assert (loaded.low, loaded.seen, loaded.high) == (7, {8, 9}, 9)
    assert not (tmp_path / "cursor.json.tmp").exists()

    path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remove it"):
        EventCursor.load(path)


def test_probability_fits_numeric_5_4() -> None:
    assert to_probability(0.123456) == Decimal("0.1235")
    assert to_probability(0.99996) == Decimal("1.0000")
    assert to_probability(1e-9) == Decimal("0.0000")
    assert to_probability(1.0000001) == Decimal("1")

    with pytest.raises(ValueError):
        to_probability(float("nan"))


def test_shap_contract() -> None:
    top = [
        {
            "feature": f"f{idx}",
            "feature_value": float("nan") if idx == 11 else idx,
            "shap_raw": 0.01 * idx * (-1) ** idx,
            "risk_direction": "neutral",
            "probability_delta_from_component": 0.001 * idx,
        }
        for idx in range(12)
    ]
    top.append(
        {
            "feature": "тип_датчика",
            "feature_value": "null",
            "shap_raw": -0.5,
            "risk_direction": "increases_probability",
            "probability_delta_from_component": 0.2,
        }
    )
    shap = {
        "base_value": 1.5,
        "values": {},
        "top": top,
        "reconstructed_raw_prediction": 2.0,
        "calculation_ms": 3.0,
    }

    result = shap_contract(shap, frozenset({"тип_датчика"}))

    assert set(result) == {"base_value", "top"}
    assert len(result["top"]) == 10
    assert result["top"][0] == {
        "feature": "тип_датчика",
        "feature_value": None,
        "shap_raw": -0.5,
        "risk_direction": "increases_probability",
        "probability_delta": 0.2,
    }
    assert [item["feature"] for item in result["top"][1:3]] == ["f11", "f10"]
    # NaN числовой фичи — тоже пропуск; значения остальных как есть.
    assert result["top"][1]["feature_value"] is None
    assert result["top"][2]["feature_value"] == 10


def test_health(tmp_path: Path) -> None:
    assert "startup" in check(tmp_path, 120)

    (tmp_path / "ready").touch()
    assert "heartbeat" in check(tmp_path, 120)

    (tmp_path / "heartbeat").touch()
    os.utime(tmp_path / "heartbeat", (1000, 1000))
    assert check(tmp_path, 120, now=1100) is None
    assert "old" in check(tmp_path, 120, now=1200)
