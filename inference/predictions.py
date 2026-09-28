"""Строки prediction_log из ответа PredictionService.predict (001, 014)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable, Mapping

import psycopg
from psycopg.types.json import Jsonb

SHAP_TOP = 10

INSERT_PREDICTION = """
    INSERT INTO prediction_log (target_kind, target_ref, incident_type,
        probability, horizon_until, model_version, shap)
    VALUES ('channel', %s, %s, %s, %s, %s, %s)
"""

# Только строки этой модели: мок и чужие версии не трогаем. Строки с
# решением диспетчера остаются — FK decisions_on_prediction без ON DELETE.
DELETE_EXPIRED = """
    DELETE FROM prediction_log p
    WHERE p.horizon_until < now()
      AND (starts_with(p.model_version, %s) OR p.model_version = %s)
      AND NOT EXISTS (
          SELECT 1 FROM decisions_on_prediction d
          WHERE d.prediction_id = p.prediction_id
      )
"""

PROBABILITY_STEP = Decimal("0.0001")


@dataclass(slots=True, frozen=True)
class PredictionRow:
    channel_id: int
    probability: Decimal
    horizon_until: datetime
    shap: dict[str, Any] | None
    incident_type: str = "CHANNEL_EVENT"


def to_probability(value: float) -> Decimal:
    """numeric(5,4) с CHECK 0..1: округление и зажим в диапазон."""
    if not math.isfinite(value):
        raise ValueError(f"Probability is not finite: {value!r}")

    rounded = Decimal(repr(value)).quantize(PROBABILITY_STEP, ROUND_HALF_UP)
    return min(max(rounded, Decimal(0)), Decimal(1))


def _plain(value: Any) -> Any:
    # numpy-скаляры в JSON не сериализуются; NaN/inf — это пропуск.
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def shap_contract(
    shap: Mapping[str, Any],
    cat_features: frozenset[str],
    top_n: int = SHAP_TOP,
) -> dict[str, Any]:
    """Выход _explain → формат prediction_log.shap из 014.

    values, reconstructed_raw_prediction и calculation_ms отбрасываются,
    строка "null" у категориальной фичи становится JSON null.
    """
    top = sorted(
        shap["top"],
        key=lambda item: abs(float(item["shap_raw"])),
        reverse=True,
    )[:top_n]

    items = []
    for item in top:
        value = _plain(item["feature_value"])
        if item["feature"] in cat_features and value == "null":
            value = None

        items.append(
            {
                "feature": item["feature"],
                "feature_value": value,
                "shap_raw": float(item["shap_raw"]),
                "risk_direction": item["risk_direction"],
                "probability_delta": float(
                    item["probability_delta_from_component"]
                ),
            }
        )

    return {"base_value": float(shap["base_value"]), "top": items}


def write_predictions(
    conn: psycopg.Connection,
    rows: Iterable[PredictionRow],
    model_version: str,
) -> None:
    params = [
        (
            str(row.channel_id),
            row.incident_type,
            row.probability,
            row.horizon_until,
            model_version,
            None if row.shap is None else Jsonb(row.shap),
        )
        for row in rows
    ]

    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(INSERT_PREDICTION, params)


def delete_expired(
    conn: psycopg.Connection,
    model_version: str,
    prediction_kind: str = "EQUIPMENT",
) -> int:
    prefix = ("incident4-v4-" if prediction_kind == "INCIDENT"
              else "catboost-aft-")
    with conn.cursor() as cur:
        cur.execute(DELETE_EXPIRED, (prefix, model_version))
        return cur.rowcount
