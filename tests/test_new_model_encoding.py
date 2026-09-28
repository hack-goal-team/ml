from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from app import DATE_COL, ID_COL, TIME_COL, VALUE_COL, PredictionService, ServiceConfig


MODEL = Path(__file__).resolve().parents[1] / "data" / "best_model.cbm"


class EmptyWeather:
    def get(self, target_hour):
        return self.fields

    def close(self):
        pass


def test_exported_model_uses_training_encoding_and_warmed_window(tmp_path: Path) -> None:
    weather = EmptyWeather()

    def load_metadata(compiled):
        weather.fields = {
            spec.name: None for spec in compiled.specs if spec.kind == "weather"
        }
        return {1: {"тип_инж_системы": "Пожарная охрана", "родитель": 7}}

    service = PredictionService(
        replace(ServiceConfig(), model_path=MODEL, log_path=tmp_path / "service.log"),
        metadata_loader=load_metadata,
        weather_client=weather,
    )
    try:
        assert len(service.feature_names) == 105
        assert service.compiled.static_features == ("тип_инж_системы", "родитель")
        assert service.store.max_window_seconds == 72 * 3600

        service.warmup([{
            ID_COL: 1, DATE_COL: "2026-09-27", TIME_COL: "10:00:00", VALUE_COL: "Норма"
        }])
        result = service.predict({
            ID_COL: 1, DATE_COL: "2026-09-28", TIME_COL: "10:00:00",
            VALUE_COL: "Норма", "horizon_hours": 24,
        })
        row = service._build_feature_row(1, datetime(2026, 9, 28, 10))
        assert row[:7] == [0, 0, 0, 0, 0, 1, 0]
        assert row[service.feature_names.index("value_cat__Норма__sum_3d")] == 2
        assert math.isfinite(result["probability"])
        assert 0 <= result["probability"] <= 1
    finally:
        service.close()


def test_encoded_model_rejects_missing_encoding_file(tmp_path: Path) -> None:
    model = tmp_path / "best_model.cbm"
    model.write_bytes(MODEL.read_bytes())
    with pytest.raises(ValueError, match="No encoding"):
        PredictionService(
            replace(ServiceConfig(), model_path=model, log_path=tmp_path / "service.log"),
            metadata_loader=lambda compiled: {},
        )
