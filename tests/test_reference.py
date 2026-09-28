from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
from catboost import CatBoostRegressor

from app import ServiceConfig, _build_metadata, compile_model_features
from inference.reference import (
    CHANNEL_COLUMNS,
    OBJECT_COLUMNS,
    build_metadata,
    fetch_metadata,
    metadata_loader,
)

DATA = Path(__file__).resolve().parent.parent / "data"
SENSORS = DATA / "справочник_каналов_датчиков.parquet"
OBJECTS = DATA / "справочник_объектов_диспетчер.parquet"
ENCODING_LEVELS = json.loads(
    (DATA / "feature_encoding.json").read_text(encoding="utf-8")
)["levels"]

# Все колонки справочников, которые можно сделать фичей, кроме ID.
ALL_FEATURES = [
    name
    for name in [*CHANNEL_COLUMNS, *OBJECT_COLUMNS]
    if name != "ид_канала_данных"
]


@pytest.fixture(scope="module")
def model_features() -> tuple[str, ...]:
    model = CatBoostRegressor()
    model.load_model(str(DATA / "best_model.cbm"))
    return tuple(model.feature_names_)


def _parquet_metadata(features, sensors=SENSORS, objects=OBJECTS):
    return _build_metadata(
        compile_model_features(features, ENCODING_LEVELS),
        ServiceConfig(
            sensors_metadata_path=sensors,
            objects_metadata_path=objects,
        ),
    )


def _registry_rows(sensors=SENSORS, objects=OBJECTS):
    # Строки parquet под именами колонок dim_channels / dim_objects.
    channels = pl.read_parquet(sensors).rename(
        CHANNEL_COLUMNS, strict=False
    )
    objs = pl.read_parquet(objects).rename(
        {"ид_объект": "object_id", **OBJECT_COLUMNS}, strict=False
    )
    return channels.to_dicts(), objs.to_dicts()


def _assert_identical(actual, expected) -> None:
    assert actual.keys() == expected.keys()

    for channel_id, features in expected.items():
        assert actual[channel_id] == features, channel_id
        for name, value in features.items():
            # 'родитель' — числовая фича: int и str для CatBoost не одно.
            assert type(actual[channel_id][name]) is type(value), (
                channel_id,
                name,
            )


def test_model_static_features_are_known(model_features) -> None:
    compiled = compile_model_features(model_features, ENCODING_LEVELS)

    assert compiled.static_features == (
        "тип_инж_системы",
        "родитель",
    )


def test_equivalent_to_parquet_for_model(model_features) -> None:
    static = compile_model_features(model_features, ENCODING_LEVELS).static_features
    expected = _parquet_metadata(model_features)
    channels, objects = _registry_rows()

    actual = build_metadata(channels, objects, static)

    assert len(expected) == 11485
    _assert_identical(actual, expected)


def test_equivalent_to_parquet_for_all_columns() -> None:
    expected = _parquet_metadata(ALL_FEATURES)
    channels, objects = _registry_rows()

    _assert_identical(
        build_metadata(channels, objects, ALL_FEATURES),
        expected,
    )


def test_edge_cases_match_parquet_path(tmp_path: Path) -> None:
    # Канал без объекта, висячая ссылка, объект без родителя, тег без '-'.
    sensors = pl.DataFrame(
        {
            "ид_канала_данных": [1, 2, 3, 4],
            "тип_датчика": ["A", "B", "C", "D"],
            "тег_инженерной_системы": ["15-11.1", "7", "-x", "1-2-3"],
            "ид_объект": [10, None, 99, 11],
        },
        schema_overrides={"ид_объект": pl.Int64},
    )
    objects = pl.DataFrame(
        {
            "ид_объект": [10, 11],
            "иерархия_уровень": [3, 2],
            "родитель": [5, None],
        },
        schema_overrides={"родитель": pl.Int64},
    )
    sensors.write_parquet(tmp_path / "s.parquet")
    objects.write_parquet(tmp_path / "o.parquet")

    features = ["тип_датчика", "тег_инженерной_системы", "родитель"]
    expected = _parquet_metadata(
        features,
        tmp_path / "s.parquet",
        tmp_path / "o.parquet",
    )
    channels, objs = _registry_rows(
        tmp_path / "s.parquet",
        tmp_path / "o.parquet",
    )
    actual = build_metadata(channels, objs, features)

    _assert_identical(actual, expected)
    assert actual[2]["родитель"] is None
    assert actual[3]["родитель"] is None
    assert actual[4]["родитель"] is None
    assert actual[3]["тег_инженерной_системы"] == ""


def test_unknown_feature_rejected() -> None:
    with pytest.raises(RuntimeError, match="нет_такой"):
        build_metadata([], [], ["нет_такой"])


def _copy(conn, table, snapshot_at, rows) -> None:
    columns = ["snapshot_at", *rows[0]]
    with conn.cursor().copy(
        f"COPY {table} ({', '.join(columns)}) FROM STDIN"
    ) as copy:
        for row in rows:
            copy.write_row([snapshot_at, *row.values()])


def test_fetch_from_postgres_takes_latest_snapshot(pg, model_features) -> None:
    channels, objects = _registry_rows()
    old = datetime(2026, 8, 1, tzinfo=timezone.utc)
    new = datetime(2026, 9, 1, tzinfo=timezone.utc)
    gone = {**channels[0], "channel_id": 424242, "system_tag": "9-1"}

    with pg.admin() as conn:
        # Старая версия с другими значениями: победить должна новая.
        _copy(
            conn,
            "dim_channels",
            old,
            [{**r, "sensor_type": "OLD", "object_id": 1} for r in channels]
            + [gone],
        )
        _copy(conn, "dim_objects", old, [{**r, "parent_id": 7} for r in objects])
        _copy(conn, "dim_channels", new, channels)
        _copy(conn, "dim_objects", new, objects)

    compiled = compile_model_features(model_features, ENCODING_LEVELS)
    expected = _parquet_metadata(model_features)

    with pg.inference() as conn:
        actual = fetch_metadata(conn, compiled.static_features)

    # Как view dim_channels_current: канал из прошлой версии не пропадает.
    assert actual.pop(424242)["тип_инж_системы"] == gone["eng_system_type"]
    _assert_identical(actual, expected)
    assert len(metadata_loader(pg.inference)(compiled)) == len(expected) + 1
