#!/usr/bin/env python3
"""Export historical stand events as monthly Parquet files for retraining."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import polars as pl


REMOTE_PSQL = (
    "docker exec -i backend-postgres-1 sh -c "
    "'exec psql -q -X -v ON_ERROR_STOP=1 -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\"'"
)
SCHEMA = {
    "ид_события": pl.Int64,
    "ид_канала_данных": pl.Int64,
    "дата": pl.Date,
    "время": pl.String,
    "значение_датчика": pl.String,
    "тревожное": pl.String,
}


def months(first: date, last: date):
    current = first
    while current <= last:
        year, month = current.year, current.month
        next_month = date(year + month // 12, month % 12 + 1, 1)
        yield current, next_month
        current = next_month


def query(start: date, end: date, limit: int | None) -> str:
    # PostgreSQL stores the original local date/time as a timestamptz in MSK.
    rows = f"LIMIT {limit}" if limit is not None else ""
    return f"""COPY (
        SELECT event_id AS "ид_события",
               channel_id AS "ид_канала_данных",
               (ts AT TIME ZONE 'Europe/Moscow')::date AS "дата",
               to_char(ts AT TIME ZONE 'Europe/Moscow', 'HH24:MI:SS') AS "время",
               raw_value AS "значение_датчика",
               CASE WHEN is_alarm THEN 't' ELSE 'f' END AS "тревожное"
        FROM events
        WHERE ts >= '{start.isoformat()} 00:00:00+03'::timestamptz
          AND ts < '{end.isoformat()} 00:00:00+03'::timestamptz
        {rows}
    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);
"""


def export_month(
    start: date,
    end: date,
    output_dir: Path,
    ssh_host: str,
    password_file: Path,
    limit: int | None,
) -> dict:
    name = f"events-{start:%Y-%m}"
    destination = output_dir / f"{name}.parquet"
    if destination.exists():
        frame = pl.scan_parquet(destination)
        if frame.collect_schema().names() != list(SCHEMA):
            raise ValueError(f"Unexpected schema in existing file: {destination}")
        return {"file": destination.name, "rows": frame.select(pl.len()).collect().item(), "skipped": True}

    csv_file = output_dir / f".{name}.csv.part"
    parquet_file = output_dir / f".{name}.parquet.part"
    csv_file.unlink(missing_ok=True)
    parquet_file.unlink(missing_ok=True)

    command = [
        "sshpass", "-f", str(password_file), "ssh", "-C",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
        ssh_host, REMOTE_PSQL,
    ]
    with csv_file.open("wb") as output:
        result = subprocess.run(
            command, input=query(start, end, limit).encode(), stdout=output,
            stderr=subprocess.PIPE, check=False,
        )
    if result.returncode:
        csv_file.unlink(missing_ok=True)
        raise RuntimeError(f"Export {name} failed: {result.stderr.decode(errors='replace')}")

    frame = pl.scan_csv(csv_file, schema_overrides=SCHEMA, try_parse_dates=True)
    row_count = frame.select(pl.len()).collect().item()
    if row_count:
        frame.sink_parquet(parquet_file, compression="zstd")
        os.replace(parquet_file, destination)
    csv_file.unlink()
    return {"file": destination.name, "rows": row_count, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-month", default="2019-01")
    parser.add_argument("--through-month", default="2026-06")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--limit-per-month", type=int, help="For a small format smoke test only")
    args = parser.parse_args()

    first = date.fromisoformat(args.from_month + "-01")
    last = date.fromisoformat(args.through_month + "-01")
    if first > last:
        parser.error("--from-month must be no later than --through-month")
    if args.limit_per_month is not None and args.limit_per_month < 1:
        parser.error("--limit-per-month must be positive")
    if not args.password_file.is_file():
        parser.error("--password-file does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for start, end in months(first, last):
        result = export_month(start, end, args.output_dir, args.ssh_host, args.password_file, args.limit_per_month)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
