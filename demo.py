from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException

from app import (
    DATE_COL,
    ID_COL,
    TIME_COL,
    VALUE_COL,
    PredictionService,
    ServiceConfig,
    create_app,
)


DEFAULT_LOGS_PATH = Path(
    "data/ext-journal-2026.parquet"
)


def create_mock_weather_app(
    route_path: str,
) -> FastAPI:
    app = FastAPI(
        title="Mock Internal Weather Service"
    )

    @app.post(route_path)
    def weather(
        payload: dict[str, Any],
    ) -> dict[str, float | int]:
        if "datetime" not in payload:
            raise HTTPException(
                status_code=422,
                detail="datetime is required",
            )

        target_hour = datetime.fromisoformat(
            str(payload["datetime"])
        )

        hour = target_hour.hour
        day = target_hour.timetuple().tm_yday

        return {
            "temperature_2m (°C)": 8.0 + 0.25 * hour,
            "relative_humidity_2m (%)": 60 + hour % 25,
            "precipitation_probability (%)": (hour * 7) % 100,
            "precipitation (mm)": 0.2 if hour % 6 == 0 else 0.0,
            "rain (mm)": 0.2 if hour % 6 == 0 else 0.0,
            "snowfall (cm)": 0.0,
            "snow_depth (m)": 0.0,
            "weather_code (wmo code)": 3,
            "cloud_cover (%)": (hour * 11) % 100,
            "pressure_msl (hPa)": 1000.0 + day % 20,
        }

    return app


def start_uvicorn(
    app: FastAPI,
    host: str,
    port: int,
) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
        )
    )

    thread = threading.Thread(
        target=server.run,
        daemon=True,
    )
    thread.start()

    return server, thread


def wait_http(
    url: str,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                url,
                timeout=1,
            ):
                return
        except Exception:
            time.sleep(0.1)

    raise RuntimeError(
        f"Service did not start: {url}"
    )


def post_json(
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        return json.loads(
            response.read()
        )


def read_real_logs(
    parquet_path: Path,
    n_predictions: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    max_date = (
        pl.scan_parquet(parquet_path)
        .select(
            pl.col(DATE_COL)
            .cast(pl.Date)
            .max()
        )
        .collect()
        .item()
    )

    if max_date is None:
        raise RuntimeError(
            "Logs parquet is empty"
        )

    start_date = (
        max_date
        - timedelta(days=4)
    )

    df = (
        pl.scan_parquet(parquet_path)
        .select(
            ID_COL,
            DATE_COL,
            TIME_COL,
            VALUE_COL,
        )
        .with_columns(
            pl.col(ID_COL).cast(pl.Int64),
            pl.col(DATE_COL).cast(pl.Date),
            pl.col(TIME_COL).cast(pl.String),
        )
        .filter(
            pl.col(DATE_COL)
            >= pl.lit(start_date)
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col(DATE_COL)
                    .cast(pl.String),
                    pl.col(TIME_COL),
                ],
                separator=" ",
            )
            .str.to_datetime(
                "%Y-%m-%d %H:%M:%S"
            )
            .alias("__datetime")
        )
        .sort(
            [
                "__datetime",
                ID_COL,
            ]
        )
        .collect()
    )

    if df.height <= n_predictions:
        raise RuntimeError(
            "Not enough recent rows for "
            "warmup + inference"
        )

    inference_start = (
        df["__datetime"].min()
        + timedelta(days=3)
    )

    inference_df = (
        df
        .filter(
            pl.col("__datetime")
            >= pl.lit(inference_start)
        )
        .head(n_predictions)
    )

    if inference_df.height < n_predictions:
        raise RuntimeError(
            "Not enough rows after "
            "3-day warmup"
        )

    first_inference_dt = (
        inference_df[
            "__datetime"
        ].min()
    )

    warmup_df = (
        df
        .filter(
            pl.col("__datetime")
            < pl.lit(
                first_inference_dt
            )
        )
        .filter(
            pl.col("__datetime")
            > pl.lit(
                first_inference_dt
                - timedelta(days=3)
            )
        )
    )

    def to_payloads(
        frame: pl.DataFrame,
    ) -> list[dict[str, Any]]:
        result = []

        for row in frame.iter_rows(
            named=True
        ):
            result.append(
                {
                    ID_COL: int(
                        row[ID_COL]
                    ),
                    DATE_COL: (
                        row[DATE_COL]
                        .isoformat()
                    ),
                    TIME_COL: (
                        row[TIME_COL]
                    ),
                    VALUE_COL: (
                        row[VALUE_COL]
                    ),
                }
            )

        return result

    return (
        to_payloads(warmup_df),
        to_payloads(inference_df),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config.yml"
        ),
    )

    parser.add_argument(
        "--logs",
        type=Path,
        default=DEFAULT_LOGS_PATH,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=24.0,
        help=(
            "Prediction horizon in hours "
            "sent to every /predict request"
        ),
    )

    args = parser.parse_args()

    if args.horizon_hours <= 0:
        parser.error(
            "--horizon-hours must be > 0"
        )

    config = ServiceConfig.from_yaml(
        args.config
    )

    weather_url = urlparse(
        config.weather_service_url
    )

    if weather_url.scheme != "http":
        raise RuntimeError(
            "Demo expects an http:// "
            "weather service URL"
        )

    weather_host = (
        weather_url.hostname
        or "127.0.0.1"
    )

    weather_port = (
        weather_url.port
        or 80
    )

    weather_path = (
        weather_url.path
        or "/weather"
    )

    mock_weather_app = (
        create_mock_weather_app(
            weather_path
        )
    )

    weather_server, weather_thread = (
        start_uvicorn(
            app=mock_weather_app,
            host=weather_host,
            port=weather_port,
        )
    )

    demo_config = replace(
        config,
        metadata_cache_path=Path(
            "runtime/demo_metadata_cache.pkl"
        ),
        state_path=Path(
            "runtime/demo_feature_state.pkl"
        ),
        log_path=Path(
            "runtime/demo_service.log"
        ),
    )

    if demo_config.state_path.exists():
        demo_config.state_path.unlink()

    warmup_logs, inference_logs = (
        read_real_logs(
            parquet_path=args.logs,
            n_predictions=args.n,
        )
    )

    service = PredictionService(
        config=demo_config
    )

    service.warmup(
        warmup_logs
    )

    prediction_app = create_app(
        service
    )

    prediction_server, prediction_thread = (
        start_uvicorn(
            app=prediction_app,
            host=config.api_host,
            port=config.api_port,
        )
    )

    api_url = (
        f"http://127.0.0.1:"
        f"{config.api_port}"
    )

    try:
        wait_http(
            f"{api_url}/health"
        )

        print(
            f"Warmup rows: "
            f"{len(warmup_logs):,}"
        )

        print(
            f"Inference rows: "
            f"{len(inference_logs):,}"
        )

        started = (
            time.perf_counter()
        )

        examples = {}

        for index, log in enumerate(
            inference_logs,
            start=1,
        ):
            payload = dict(log)
            payload[
                "horizon_hours"
            ] = args.horizon_hours

            if index == 1:
                payload[
                    "add_shap"
                ] = "all"
            elif index == 2:
                payload[
                    "add_shap"
                ] = "auto"
            else:
                payload[
                    "add_shap"
                ] = "none"

            response = post_json(
                f"{api_url}/predict",
                payload,
            )

            if index <= 3:
                examples[index] = (
                    response
                )

            if index % 100 == 0:
                print(
                    f"{index:,}/"
                    f"{len(inference_logs):,}"
                )

        elapsed = (
            time.perf_counter()
            - started
        )

        print()
        print(
            "Example add_shap='all':"
        )
        print(
            json.dumps(
                examples[1],
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            "Example add_shap='auto':"
        )
        print(
            json.dumps(
                examples[2],
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            "Example add_shap='none':"
        )
        print(
            json.dumps(
                examples[3],
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            f"Requests: "
            f"{len(inference_logs):,}"
        )
        print(
            f"Wall time: "
            f"{elapsed:.3f} s"
        )
        print(
            "Average HTTP latency: "
            f"{elapsed / len(inference_logs) * 1000:.3f} ms"
        )
        print(
            "RPS: "
            f"{len(inference_logs) / elapsed:.1f}"
        )

    finally:
        prediction_server.should_exit = True
        weather_server.should_exit = True

        prediction_thread.join(
            timeout=10
        )
        weather_thread.join(
            timeout=10
        )


if __name__ == "__main__":
    main()
