from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

SCHEMA_PATH = Path(__file__).with_name("test_db_schema.sql")
TEMPLATE_DB = "inference_template"

_db_numbers = itertools.count()


@dataclass(frozen=True)
class PgDatabase:
    params: dict[str, Any]

    def admin(self) -> psycopg.Connection:
        return psycopg.connect(**self.params, autocommit=True)

    def inference(self) -> psycopg.Connection:
        # Под ролью с грантами из 004: SQL обязан работать без суперюзера.
        return psycopg.connect(
            **{**self.params, "user": "inference"},
            autocommit=True,
        )


@pytest.fixture(scope="session")
def pg_server_params(tmp_path_factory) -> Iterator[dict[str, Any]]:
    # Встроенный Postgres без Docker; нет пакета — SQL-тесты пропускаются.
    pgserver = pytest.importorskip(
        "pgserver",
        reason="pgserver not installed: pip install -r requirements-dev.txt",
    )

    try:
        server = pgserver.get_server(
            tmp_path_factory.mktemp("pgdata"),
            cleanup_mode="stop",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedded Postgres failed to start: {exc}")

    params = conninfo_to_dict(server.get_uri())

    with psycopg.connect(**params, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {TEMPLATE_DB}")

    with psycopg.connect(
        **{**params, "dbname": TEMPLATE_DB},
        autocommit=True,
    ) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    yield params

    server.cleanup()


@pytest.fixture
def pg(pg_server_params) -> PgDatabase:
    # Своя база на тест: копия шаблона со схемой, без чистки между тестами.
    name = f"inference_test_{next(_db_numbers)}"

    with psycopg.connect(**pg_server_params, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}")

    return PgDatabase({**pg_server_params, "dbname": name})
