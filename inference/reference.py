"""Static-фичи модели из справочников Postgres вместо parquet.

Результат совпадает с app._build_metadata: те же ключи, типы и значения.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

from app import ID_COL, CompiledModelFeatures

# Колонка parquet (имя фичи) → колонка dim_channels (001, 012).
CHANNEL_COLUMNS = {
    ID_COL: "channel_id",
    "тип_инж_системы": "eng_system_type",
    "тип_датчика": "sensor_type",
    "тег_инженерной_системы": "system_tag",
    "название_датчика": "sensor_name",
    "ид_объект": "object_id",
}

# Колонка parquet → колонка dim_objects. Как и в app.py, при совпадении
# имени приоритет у справочника каналов.
OBJECT_COLUMNS = {
    "иерархия_уровень": "hierarchy_level",
    "родитель": "parent_id",
    "вид_объекта": "object_kind",
    "диспетчерское_название_объекта": "dispatcher_name",
}

TAG_FEATURE = "тег_инженерной_системы"

# Последняя версия каждой строки — та же семантика, что у view
# dim_*_current (ADR-031). Сами view роли inference не выданы (004).
SELECT_CHANNELS = """
    SELECT DISTINCT ON (channel_id)
        channel_id, eng_system_type, sensor_type, system_tag,
        sensor_name, object_id
    FROM dim_channels
    ORDER BY channel_id, snapshot_at DESC
"""

SELECT_OBJECTS = """
    SELECT DISTINCT ON (object_id)
        object_id, hierarchy_level, parent_id, object_kind,
        dispatcher_name
    FROM dim_objects
    ORDER BY object_id, snapshot_at DESC
"""


def build_metadata(
    channels: Iterable[Mapping[str, Any]],
    objects: Iterable[Mapping[str, Any]],
    static_features: Sequence[str],
) -> dict[int, dict[str, Any]]:
    """Чистое преобразование строк справочников в metadata модели."""
    missing = [
        feature
        for feature in static_features
        if feature not in CHANNEL_COLUMNS
        and feature not in OBJECT_COLUMNS
    ]

    if missing:
        raise RuntimeError(
            "Registry does not contain model features: "
            f"{missing}"
        )

    objects_by_id = {
        row["object_id"]: row
        for row in objects
    }

    metadata: dict[int, dict[str, Any]] = {}

    for channel in channels:
        # Left join, как в app.py: канал без объекта или с висячей
        # ссылкой остаётся, объектные фичи у него None.
        obj = objects_by_id.get(channel["object_id"])
        features: dict[str, Any] = {}

        for feature in static_features:
            if feature in CHANNEL_COLUMNS:
                value = channel[CHANNEL_COLUMNS[feature]]

                # Префикс до первого '-', как str.split('-').list.get(0).
                if feature == TAG_FEATURE and value is not None:
                    value = str(value).split("-")[0]
            elif obj is not None:
                value = obj[OBJECT_COLUMNS[feature]]
            else:
                value = None

            features[feature] = value

        metadata[int(channel["channel_id"])] = features

    return metadata


def fetch_metadata(
    conn: psycopg.Connection,
    static_features: Sequence[str],
) -> dict[int, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SELECT_CHANNELS)
        channels = cur.fetchall()

        cur.execute(SELECT_OBJECTS)
        objects = cur.fetchall()

    return build_metadata(
        channels,
        objects,
        static_features,
    )


def metadata_loader(
    connect: Callable[[], psycopg.Connection],
) -> Callable[[CompiledModelFeatures], dict[int, dict[str, Any]]]:
    """Загрузчик для PredictionService(metadata_loader=...)."""

    def load(
        compiled: CompiledModelFeatures,
    ) -> dict[int, dict[str, Any]]:
        with connect() as conn:
            return fetch_metadata(
                conn,
                compiled.static_features,
            )

    return load
