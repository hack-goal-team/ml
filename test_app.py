from __future__ import annotations

import json
import math
import random
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from app import (
    OnlineFeatureStore,
    PredictionRequestError,
    PredictionService,
    ServiceConfig,
    WeatherCache,
    WeatherServiceClient,
    compile_model_features,
    normalize_horizon_hours,
    normalize_shap_mode,
    should_calculate_shap,
)


def test_config_from_yaml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"

    config_path.write_text(
        """
api:
  host: 127.0.0.1
  port: 9000

weather_service:
  url: http://127.0.0.1:9010/weather
  timeout_seconds: 2
  timezone: Europe/Moscow
  cache_max_entries: 32

shap:
  auto_threshold: 0.7
""",
        encoding="utf-8",
    )

    config = ServiceConfig.from_yaml(
        config_path
    )

    assert config.api_host == "127.0.0.1"
    assert config.api_port == 9000
    assert (
        config.weather_service_url
        == "http://127.0.0.1:9010/weather"
    )
    assert config.weather_timeout_seconds == 2
    assert config.weather_timezone == "Europe/Moscow"
    assert config.weather_cache_max_entries == 32
    assert config.shap_auto_threshold == 0.7


def test_shap_modes() -> None:
    assert normalize_shap_mode("none") == "none"
    assert normalize_shap_mode("ALL") == "all"
    assert normalize_shap_mode("Auto") == "auto"

    assert not should_calculate_shap(
        "none",
        probability=0.99,
        threshold=0.50,
    )

    assert should_calculate_shap(
        "all",
        probability=0.01,
        threshold=0.50,
    )

    assert not should_calculate_shap(
        "auto",
        probability=0.50,
        threshold=0.50,
    )

    assert should_calculate_shap(
        "auto",
        probability=0.51,
        threshold=0.50,
    )


def test_horizon_hours_validation() -> None:
    assert normalize_horizon_hours(24) == 24.0
    assert normalize_horizon_hours(6.5) == 6.5
    assert normalize_horizon_hours("48") == 48.0

    for value in (0, -1, float("inf"), float("nan"), True, "x"):
        try:
            normalize_horizon_hours(value)
        except PredictionRequestError:
            pass
        else:
            raise AssertionError(
                f"Expected invalid horizon: {value!r}"
            )


def test_probability_depends_on_horizon() -> None:
    service = PredictionService.__new__(
        PredictionService
    )
    service.scale = 1.0

    raw_prediction = math.log(24.0)

    p_6h = service._probability_from_raw(
        raw_prediction,
        6.0,
    )
    p_24h = service._probability_from_raw(
        raw_prediction,
        24.0,
    )
    p_48h = service._probability_from_raw(
        raw_prediction,
        48.0,
    )

    assert p_6h < p_24h < p_48h
    assert math.isclose(
        p_24h,
        0.5,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_weather_http_contract_and_cache() -> None:
    received = []

    class Handler(
        BaseHTTPRequestHandler
    ):
        def do_POST(self) -> None:
            length = int(
                self.headers[
                    "Content-Length"
                ]
            )

            payload = json.loads(
                self.rfile.read(
                    length
                )
            )

            received.append(
                {
                    "path": self.path,
                    "payload": payload,
                }
            )

            response = {
                "temperature_2m (°C)": 12.4,
                "relative_humidity_2m (%)": 71,
                "precipitation_probability (%)": 20,
                "precipitation (mm)": 0.0,
                "rain (mm)": 0.0,
                "snowfall (cm)": 0.0,
                "snow_depth (m)": 0.0,
                "weather_code (wmo code)": 3,
                "cloud_cover (%)": 85,
                "pressure_msl (hPa)": 1012.4,
            }

            body = json.dumps(
                response
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(
            self,
            format,
            *args,
        ) -> None:
            return

    server = HTTPServer(
        ("127.0.0.1", 0),
        Handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    port = server.server_address[1]

    client = WeatherServiceClient(
        url=(
            f"http://127.0.0.1:"
            f"{port}/weather"
        ),
        timeout_seconds=2,
        timezone="Europe/Moscow",
    )

    cache = WeatherCache(
        client=client,
        max_entries=10,
    )

    target = datetime(
        2026,
        9,
        22,
        14,
        37,
        15,
    )

    first = cache.get(target)
    second = cache.get(target)

    assert first == second
    assert first[
        "temperature_2m (°C)"
    ] == 12.4

    assert len(received) == 1

    assert received[0][
        "path"
    ] == "/weather"

    assert received[0][
        "payload"
    ] == {
        "datetime": (
            "2026-09-22T14:00:00"
        ),
        "timezone": "Europe/Moscow",
    }

    client.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_closed_right_window() -> None:
    compiled = compile_model_features(
        [
            "тип_датчика",
            "value_cat__Норма__sum_1h",
        ]
    )

    store = OnlineFeatureStore(
        compiled=compiled,
        metadata={
            1: {
                "тип_датчика": "A",
            }
        },
    )

    t0 = datetime(
        2026,
        1,
        1,
        12,
        0,
    )

    store.update(
        sensor_id=1,
        timestamp=t0,
        value="Норма",
    )

    store.advance(
        sensor_id=1,
        at=t0 + timedelta(hours=1),
    )

    state = store.get_state(1)

    assert (
        state.category_counts[3600]
        .get("Норма", 0)
        == 0
    )


def _brute_force_count(
    history,
    at,
    window_seconds,
    category,
):
    cutoff = (
        at
        - timedelta(
            seconds=window_seconds
        )
    )

    return sum(
        1
        for timestamp, value
        in history
        if (
            cutoff < timestamp <= at
            and value == category
        )
    )


def _brute_force_mean(
    history,
    at,
    window_seconds,
):
    cutoff = (
        at
        - timedelta(
            seconds=window_seconds
        )
    )

    values = [
        float(value)
        for timestamp, value
        in history
        if (
            cutoff < timestamp <= at
            and isinstance(
                value,
                (int, float),
            )
        )
    ]

    if not values:
        return None

    return sum(values) / len(values)


def test_incremental_equals_bruteforce() -> None:
    compiled = compile_model_features(
        [
            "тип_датчика",
            "value_cat__Норма__sum_1h",
            "value_cat__Неисправен__sum_6h",
            "value_float__mean_2h",
        ]
    )

    store = OnlineFeatureStore(
        compiled=compiled,
        metadata={
            1: {
                "тип_датчика": "A",
            }
        },
    )

    rng = random.Random(42)

    now = datetime(
        2026,
        1,
        1,
        0,
        0,
    )

    history = []

    values = [
        "Норма",
        "Неисправен",
        10.0,
        20.0,
        30.0,
    ]

    for _ in range(500):
        now += timedelta(
            seconds=rng.randint(
                1,
                900,
            )
        )

        value = rng.choice(
            values
        )

        history.append(
            (
                now,
                value,
            )
        )

        store.update(
            sensor_id=1,
            timestamp=now,
            value=value,
        )

        state = store.get_state(1)

        expected_norm = (
            _brute_force_count(
                history,
                now,
                3600,
                "Норма",
            )
        )

        expected_fault = (
            _brute_force_count(
                history,
                now,
                6 * 3600,
                "Неисправен",
            )
        )

        expected_mean = (
            _brute_force_mean(
                history,
                now,
                2 * 3600,
            )
        )

        assert (
            state
            .category_counts[
                3600
            ]
            .get(
                "Норма",
                0,
            )
            == expected_norm
        )

        assert (
            state
            .category_counts[
                6 * 3600
            ]
            .get(
                "Неисправен",
                0,
            )
            == expected_fault
        )

        count = (
            state
            .numeric_count[
                2 * 3600
            ]
        )

        if expected_mean is None:
            assert count == 0
        else:
            actual = (
                state
                .numeric_sum[
                    2 * 3600
                ]
                / count
            )

            assert math.isclose(
                actual,
                expected_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
