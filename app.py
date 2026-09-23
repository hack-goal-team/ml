from __future__ import annotations

import argparse
import logging
import math
import pickle
import re
import threading
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import polars as pl
import requests
import uvicorn
import yaml
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException


ID_COL = "ид_канала_данных"
DATE_COL = "дата"
TIME_COL = "время"
VALUE_COL = "значение_датчика"

HORIZON_HOURS_FIELD = "horizon_hours"

ShapMode = Literal["none", "all", "auto"]


class PredictionRequestError(ValueError):
    pass


@dataclass(slots=True)
class ServiceConfig:
    # Адрес, на котором поднимается prediction API.
    api_host: str = "0.0.0.0"

    # Порт prediction API.
    api_port: int = 8000

    # Путь к обученной CatBoost SurvivalAft модели.
    model_path: Path = Path("survival_optuna/best_model.cbm")

    # Справочник каналов датчиков.
    sensors_metadata_path: Path = Path(
        "data/справочник_каналов_датчиков.parquet"
    )

    # Справочник объектов.
    objects_metadata_path: Path = Path(
        "data/справочник_объектов_диспетчер.parquet"
    )

    # Кеш подготовленной metadata.
    metadata_cache_path: Path = Path("runtime/metadata_cache.pkl")

    # Snapshot rolling-state для восстановления после рестарта.
    state_path: Path = Path("runtime/feature_state.pkl")

    # Файл runtime-логов.
    log_path: Path = Path("runtime/service.log")

    # Порог probability, выше которого auto включает SHAP.
    shap_auto_threshold: float = 0.50

    # Автосохранение state каждые N прогнозов; 0 отключает.
    state_save_every_n: int = 0

    # URL внутреннего погодного сервиса.
    weather_service_url: str = "http://127.0.0.1:8010/weather"

    # Максимальное ожидание ответа погодного сервиса, секунд.
    weather_timeout_seconds: float = 5.0

    # Таймзона, передаваемая погодному сервису.
    weather_timezone: str = "Europe/Moscow"

    # Максимальное число погодных часов в RAM-кеше.
    weather_cache_max_entries: int = 256

    # Максимальный размер одного log-файла до ротации.
    log_max_bytes: int = 50 * 1024 * 1024

    # Число старых log-файлов после ротации.
    log_backup_count: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ServiceConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        defaults = cls()

        api = data.get("api", {})
        model = data.get("model", {})
        metadata = data.get("metadata", {})
        runtime = data.get("runtime", {})
        shap = data.get("shap", {})
        weather = data.get("weather_service", {})
        logging_cfg = data.get("logging", {})

        return cls(
            api_host=str(api.get("host", defaults.api_host)),
            api_port=int(api.get("port", defaults.api_port)),
            model_path=Path(model.get("path", defaults.model_path)),
            sensors_metadata_path=Path(
                metadata.get(
                    "sensors_path",
                    defaults.sensors_metadata_path,
                )
            ),
            objects_metadata_path=Path(
                metadata.get(
                    "objects_path",
                    defaults.objects_metadata_path,
                )
            ),
            metadata_cache_path=Path(
                runtime.get(
                    "metadata_cache_path",
                    defaults.metadata_cache_path,
                )
            ),
            state_path=Path(
                runtime.get("state_path", defaults.state_path)
            ),
            log_path=Path(
                runtime.get("log_path", defaults.log_path)
            ),
            shap_auto_threshold=float(
                shap.get(
                    "auto_threshold",
                    defaults.shap_auto_threshold,
                )
            ),
            state_save_every_n=int(
                runtime.get(
                    "state_save_every_n",
                    defaults.state_save_every_n,
                )
            ),
            weather_service_url=str(
                weather.get(
                    "url",
                    defaults.weather_service_url,
                )
            ),
            weather_timeout_seconds=float(
                weather.get(
                    "timeout_seconds",
                    defaults.weather_timeout_seconds,
                )
            ),
            weather_timezone=str(
                weather.get(
                    "timezone",
                    defaults.weather_timezone,
                )
            ),
            weather_cache_max_entries=int(
                weather.get(
                    "cache_max_entries",
                    defaults.weather_cache_max_entries,
                )
            ),
            log_max_bytes=int(
                logging_cfg.get(
                    "max_bytes",
                    defaults.log_max_bytes,
                )
            ),
            log_backup_count=int(
                logging_cfg.get(
                    "backup_count",
                    defaults.log_backup_count,
                )
            ),
        )


def configure_logging(config: ServiceConfig) -> logging.Logger:
    logger = logging.getLogger("sensor_prediction_service")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    config.log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_handler = RotatingFileHandler(
        config.log_path,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class WeatherServiceClient:
    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        timezone: str,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.timezone = timezone
        self.session = requests.Session()

    def get(
        self,
        target_hour: datetime,
    ) -> Mapping[str, float | int | None]:
        target_hour = target_hour.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        payload = {
            "datetime": target_hour.isoformat(
                timespec="seconds"
            ),
            "timezone": self.timezone,
        }

        try:
            response = self.session.post(
                self.url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Weather service request failed: "
                f"url={self.url}, target_hour={target_hour}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Weather service returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Weather service response must be a JSON object"
            )

        return data

    def close(self) -> None:
        self.session.close()


class WeatherCache:
    def __init__(
        self,
        client: WeatherServiceClient,
        max_entries: int,
    ) -> None:
        self.client = client
        self.max_entries = max_entries
        self._values: OrderedDict[
            datetime,
            Mapping[str, float | int | None],
        ] = OrderedDict()

    def get(
        self,
        target_hour: datetime,
    ) -> Mapping[str, float | int | None]:
        target_hour = target_hour.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        if target_hour in self._values:
            value = self._values.pop(target_hour)
            self._values[target_hour] = value
            return value

        value = self.client.get(target_hour)
        self._values[target_hour] = value

        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

        return value


def parse_datetime(
    date_value: Any,
    time_value: Any,
) -> datetime:
    if isinstance(date_value, datetime):
        d = date_value.date()
    elif isinstance(date_value, date):
        d = date_value
    else:
        d = date.fromisoformat(str(date_value))

    if isinstance(time_value, datetime):
        t = time_value.time()
    elif isinstance(time_value, dt_time):
        t = time_value
    else:
        t = dt_time.fromisoformat(str(time_value))

    return datetime.combine(d, t)


def split_sensor_value(
    value: Any,
) -> tuple[str | None, float | None]:
    if value is None:
        return None, None

    if isinstance(value, bool):
        return str(value), None

    if isinstance(value, (int, float)):
        numeric = float(value)
        return (
            (None, None)
            if math.isnan(numeric)
            else (None, numeric)
        )

    raw = str(value)

    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return raw, None

    return (
        (None, None)
        if math.isnan(numeric)
        else (None, numeric)
    )


CAT_RE = re.compile(
    r"^value_cat__(?P<category>.+)__sum_"
    r"(?P<window>current|\d+(?:\.\d+)?[hdm])$"
)

NUM_RE = re.compile(
    r"^value_float__mean_"
    r"(?P<window>current|\d+(?:\.\d+)?[hdm])$"
)

WEATHER_RE = re.compile(
    r"^(?P<name>.+)__future_"
    r"(?P<horizon>current|\d+(?:\.\d+)?[hdm])$"
)


def period_seconds(value: str) -> int:
    if value == "current":
        return 0

    number = float(value[:-1])
    unit = value[-1]

    if unit == "d":
        return int(round(number * 24 * 3600))
    if unit == "h":
        return int(round(number * 3600))
    if unit == "m":
        return int(round(number * 60))

    raise ValueError(f"Unknown period: {value}")


@dataclass(slots=True, frozen=True)
class FeatureSpec:
    kind: str
    name: str | None = None
    seconds: int = 0


@dataclass(slots=True, frozen=True)
class CompiledModelFeatures:
    names: tuple[str, ...]
    specs: tuple[FeatureSpec, ...]
    rolling_windows: tuple[int, ...]
    categories_by_window: dict[int, frozenset[str]]
    numeric_windows: frozenset[int]
    static_features: tuple[str, ...]
    weather_offsets_seconds: tuple[int, ...]


def compile_model_features(
    feature_names: Iterable[str],
) -> CompiledModelFeatures:
    names = tuple(feature_names)
    specs: list[FeatureSpec] = []
    rolling_windows: set[int] = set()
    categories_by_window: dict[int, set[str]] = {}
    numeric_windows: set[int] = set()
    static_features: list[str] = []
    weather_offsets_seconds: set[int] = set()

    for feature in names:
        cat_match = CAT_RE.match(feature)

        if cat_match:
            category = cat_match.group("category")
            seconds = period_seconds(
                cat_match.group("window")
            )

            specs.append(
                FeatureSpec(
                    kind="category",
                    name=category,
                    seconds=seconds,
                )
            )

            if seconds > 0:
                rolling_windows.add(seconds)
                categories_by_window.setdefault(
                    seconds,
                    set(),
                ).add(category)

            continue

        num_match = NUM_RE.match(feature)

        if num_match:
            seconds = period_seconds(
                num_match.group("window")
            )

            specs.append(
                FeatureSpec(
                    kind="numeric",
                    seconds=seconds,
                )
            )

            if seconds > 0:
                rolling_windows.add(seconds)
                numeric_windows.add(seconds)

            continue

        weather_match = WEATHER_RE.match(feature)

        if weather_match:
            seconds = period_seconds(
                weather_match.group("horizon")
            )

            specs.append(
                FeatureSpec(
                    kind="weather",
                    name=weather_match.group("name"),
                    seconds=seconds,
                )
            )

            weather_offsets_seconds.add(seconds)
            continue

        if feature == "day_of_week":
            specs.append(
                FeatureSpec(kind="day_of_week")
            )
            continue

        specs.append(
            FeatureSpec(
                kind="static",
                name=feature,
            )
        )
        static_features.append(feature)

    return CompiledModelFeatures(
        names=names,
        specs=tuple(specs),
        rolling_windows=tuple(
            sorted(rolling_windows)
        ),
        categories_by_window={
            window: frozenset(categories)
            for window, categories
            in categories_by_window.items()
        },
        numeric_windows=frozenset(
            numeric_windows
        ),
        static_features=tuple(
            static_features
        ),
        weather_offsets_seconds=tuple(
            sorted(weather_offsets_seconds)
        ),
    )


def _file_signature(
    path: Path,
) -> tuple[str, int, int]:
    stat = path.stat()

    return (
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
    )


def _metadata_signature(
    compiled: CompiledModelFeatures,
    config: ServiceConfig,
) -> dict[str, Any]:
    return {
        "static_features": compiled.static_features,
        "sensors": _file_signature(
            config.sensors_metadata_path
        ),
        "objects": _file_signature(
            config.objects_metadata_path
        ),
    }


def _build_metadata(
    compiled: CompiledModelFeatures,
    config: ServiceConfig,
) -> dict[int, dict[str, Any]]:
    required = tuple(
        compiled.static_features
    )

    sensors_schema = (
        pl.scan_parquet(
            config.sensors_metadata_path
        )
        .collect_schema()
    )

    objects_schema = (
        pl.scan_parquet(
            config.objects_metadata_path
        )
        .collect_schema()
    )

    sensors_columns = set(
        sensors_schema.names()
    )

    objects_columns = set(
        objects_schema.names()
    )

    missing = [
        feature
        for feature in required
        if feature not in sensors_columns
        and feature not in objects_columns
    ]

    if missing:
        raise RuntimeError(
            "Metadata does not contain model features: "
            f"{missing}"
        )

    sensor_required = [
        feature
        for feature in required
        if feature in sensors_columns
    ]

    object_required = [
        feature
        for feature in required
        if feature not in sensor_required
        and feature in objects_columns
    ]

    sensor_select = list(
        dict.fromkeys(
            [
                ID_COL,
                "ид_объект",
                *sensor_required,
            ]
        )
    )

    object_select = list(
        dict.fromkeys(
            [
                "ид_объект",
                *object_required,
            ]
        )
    )

    sensors = (
        pl.read_parquet(
            config.sensors_metadata_path,
            columns=sensor_select,
        )
        .with_columns(
            pl.col(ID_COL).cast(pl.Int64),
            pl.col("ид_объект").cast(
                pl.Int64
            ),
        )
    )

    if (
        "тег_инженерной_системы"
        in sensors.columns
    ):
        sensors = sensors.with_columns(
            pl.col(
                "тег_инженерной_системы"
            )
            .cast(pl.String)
            .str.split("-")
            .list.get(0)
        )

    objects = (
        pl.read_parquet(
            config.objects_metadata_path,
            columns=object_select,
        )
        .with_columns(
            pl.col("ид_объект").cast(
                pl.Int64
            )
        )
    )

    metadata_df = (
        sensors
        .join(
            objects,
            on="ид_объект",
            how="left",
        )
        .select(
            ID_COL,
            *required,
        )
    )

    return {
        int(row[ID_COL]): {
            feature: row[feature]
            for feature in required
        }
        for row
        in metadata_df.iter_rows(
            named=True
        )
    }


def load_metadata(
    compiled: CompiledModelFeatures,
    config: ServiceConfig,
) -> dict[int, dict[str, Any]]:
    signature = _metadata_signature(
        compiled,
        config,
    )

    if config.metadata_cache_path.exists():
        try:
            with config.metadata_cache_path.open(
                "rb"
            ) as file:
                cache = pickle.load(file)

            if (
                cache.get("signature")
                == signature
            ):
                return cache["metadata"]
        except Exception:
            pass

    metadata = _build_metadata(
        compiled,
        config,
    )

    config.metadata_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = (
        config.metadata_cache_path
        .with_suffix(
            config.metadata_cache_path.suffix
            + ".tmp"
        )
    )

    with tmp_path.open("wb") as file:
        pickle.dump(
            {
                "signature": signature,
                "metadata": metadata,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    tmp_path.replace(
        config.metadata_cache_path
    )

    return metadata


@dataclass(slots=True, frozen=True)
class Event:
    timestamp: datetime
    category: str | None
    numeric: float | None


@dataclass(slots=True)
class SensorState:
    events: list[Event] = field(
        default_factory=list
    )

    positions: dict[int, int] = field(
        default_factory=dict
    )

    category_counts: dict[
        int,
        Counter[str],
    ] = field(
        default_factory=dict
    )

    numeric_sum: dict[int, float] = field(
        default_factory=dict
    )

    numeric_count: dict[int, int] = field(
        default_factory=dict
    )

    watermark: datetime | None = None
    latest_event: Event | None = None
    first_seen: datetime | None = None


class OnlineFeatureStore:
    def __init__(
        self,
        compiled: CompiledModelFeatures,
        metadata: dict[int, dict[str, Any]],
    ) -> None:
        self.compiled = compiled
        self.metadata = metadata
        self.states: dict[
            int,
            SensorState,
        ] = {}

    @property
    def max_window_seconds(self) -> int:
        return max(
            self.compiled.rolling_windows,
            default=0,
        )

    def _new_state(
        self,
    ) -> SensorState:
        return SensorState(
            positions={
                window: 0
                for window
                in self.compiled.rolling_windows
            },
            category_counts={
                window: Counter()
                for window
                in self.compiled.rolling_windows
            },
            numeric_sum={
                window: 0.0
                for window
                in self.compiled.numeric_windows
            },
            numeric_count={
                window: 0
                for window
                in self.compiled.numeric_windows
            },
        )

    def get_state(
        self,
        sensor_id: int,
    ) -> SensorState:
        state = self.states.get(
            sensor_id
        )

        if state is None:
            state = self._new_state()
            self.states[
                sensor_id
            ] = state

        return state

    def update(
        self,
        sensor_id: int,
        timestamp: datetime,
        value: Any,
    ) -> None:
        state = self.get_state(
            sensor_id
        )

        if (
            state.watermark is not None
            and timestamp < state.watermark
        ):
            raise ValueError(
                "Out-of-order event for "
                f"sensor {sensor_id}: "
                f"{timestamp} < {state.watermark}"
            )

        category, numeric = (
            split_sensor_value(value)
        )

        event = Event(
            timestamp=timestamp,
            category=category,
            numeric=numeric,
        )

        state.events.append(event)
        state.latest_event = event

        if state.first_seen is None:
            state.first_seen = timestamp

        for window in (
            self.compiled.rolling_windows
        ):
            tracked_categories = (
                self.compiled
                .categories_by_window
                .get(window)
            )

            if (
                category is not None
                and tracked_categories
                is not None
                and category
                in tracked_categories
            ):
                state.category_counts[
                    window
                ][category] += 1

            if (
                window
                in self.compiled.numeric_windows
                and numeric is not None
            ):
                state.numeric_sum[
                    window
                ] += numeric

                state.numeric_count[
                    window
                ] += 1

        self.advance(
            sensor_id,
            timestamp,
        )

    def advance(
        self,
        sensor_id: int,
        at: datetime,
    ) -> None:
        state = self.get_state(
            sensor_id
        )

        if (
            state.watermark is not None
            and at < state.watermark
        ):
            raise ValueError(
                "Prediction time "
                f"{at} is before "
                f"sensor watermark "
                f"{state.watermark}"
            )

        events = state.events

        for window in (
            self.compiled.rolling_windows
        ):
            cutoff = (
                at
                - timedelta(
                    seconds=window
                )
            )

            pos = state.positions[
                window
            ]

            tracked_categories = (
                self.compiled
                .categories_by_window
                .get(window)
            )

            numeric_tracked = (
                window
                in self.compiled.numeric_windows
            )

            while (
                pos < len(events)
                and events[
                    pos
                ].timestamp <= cutoff
            ):
                old = events[pos]

                if (
                    old.category
                    is not None
                    and tracked_categories
                    is not None
                    and old.category
                    in tracked_categories
                ):
                    counter = (
                        state.category_counts[
                            window
                        ]
                    )

                    counter[
                        old.category
                    ] -= 1

                    if (
                        counter[
                            old.category
                        ]
                        == 0
                    ):
                        del counter[
                            old.category
                        ]

                if (
                    numeric_tracked
                    and old.numeric
                    is not None
                ):
                    state.numeric_sum[
                        window
                    ] -= old.numeric

                    state.numeric_count[
                        window
                    ] -= 1

                pos += 1

            state.positions[
                window
            ] = pos

        state.watermark = at
        self._compact(state)

    def _compact(
        self,
        state: SensorState,
    ) -> None:
        if not (
            self.compiled.rolling_windows
        ):
            return

        removable = min(
            state.positions.values()
        )

        if removable < 1024:
            return

        if (
            removable
            < len(state.events) // 2
        ):
            return

        del state.events[
            :removable
        ]

        for window in (
            self.compiled.rolling_windows
        ):
            state.positions[
                window
            ] -= removable

    def history_complete(
        self,
        sensor_id: int,
        at: datetime,
    ) -> bool:
        state = self.get_state(
            sensor_id
        )

        if state.first_seen is None:
            return False

        return (
            at - state.first_seen
        ).total_seconds() >= (
            self.max_window_seconds
        )


def normalize_shap_mode(
    value: Any,
) -> ShapMode:
    mode = str(
        value
        if value is not None
        else "none"
    ).lower()

    if mode not in {
        "none",
        "all",
        "auto",
    }:
        raise ValueError(
            "add_shap must be "
            "'none', 'all' or 'auto'"
        )

    return mode  # type: ignore[return-value]


def normalize_horizon_hours(
    value: Any,
) -> float:
    if isinstance(value, bool):
        raise PredictionRequestError(
            "horizon_hours must be a positive number"
        )

    try:
        horizon_hours = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionRequestError(
            "horizon_hours must be a positive number"
        ) from exc

    if (
        not math.isfinite(horizon_hours)
        or horizon_hours <= 0
    ):
        raise PredictionRequestError(
            "horizon_hours must be a positive number"
        )

    return horizon_hours


def should_calculate_shap(
    mode: ShapMode,
    probability: float,
    threshold: float,
) -> bool:
    if mode == "all":
        return True

    if mode == "none":
        return False

    return probability > threshold


def _json_safe(
    value: Any,
) -> Any:
    if (
        isinstance(value, float)
        and (
            math.isnan(value)
            or math.isinf(value)
        )
    ):
        return None

    return value


class PredictionService:
    def __init__(
        self,
        config: ServiceConfig,
    ) -> None:
        self.config = config
        self.logger = configure_logging(
            config
        )

        started = time.perf_counter()

        self.model = CatBoostRegressor()
        self.model.load_model(
            str(config.model_path)
        )

        self.feature_names = tuple(
            self.model.feature_names_
        )

        self.compiled = (
            compile_model_features(
                self.feature_names
            )
        )

        self.cat_feature_indices = (
            frozenset(
                self.model
                .get_cat_feature_indices()
            )
        )

        self.metadata = load_metadata(
            compiled=self.compiled,
            config=config,
        )

        self.store = OnlineFeatureStore(
            compiled=self.compiled,
            metadata=self.metadata,
        )

        self.weather_client = (
            WeatherServiceClient(
                url=config.weather_service_url,
                timeout_seconds=(
                    config
                    .weather_timeout_seconds
                ),
                timezone=(
                    config
                    .weather_timezone
                ),
            )
        )

        self.weather_cache = (
            WeatherCache(
                client=(
                    self.weather_client
                ),
                max_entries=(
                    config
                    .weather_cache_max_entries
                ),
            )
        )

        self.scale = self._read_scale()
        self.lock = threading.Lock()
        self.n_predictions = 0

        self.load_state_if_exists()

        self.logger.info(
            "service_initialized "
            "model=%s features=%d "
            "metadata_sensors=%d "
            "state_sensors=%d "
            "weather_url=%s "
            "startup_ms=%.2f",
            config.model_path,
            len(self.feature_names),
            len(self.metadata),
            len(self.store.states),
            config.weather_service_url,
            (
                time.perf_counter()
                - started
            ) * 1000,
        )

    def _read_scale(
        self,
    ) -> float:
        loss = str(
            self.model
            .get_all_params()
            .get("loss_function")
        )

        match = re.search(
            r"(?:^|;)scale="
            r"([0-9eE+\-.]+)",
            loss,
        )

        if match is None:
            raise RuntimeError(
                "Could not read "
                "SurvivalAft scale "
                "from model"
            )

        return float(
            match.group(1)
        )

    def _probability_from_raw(
        self,
        raw_prediction: float,
        horizon_hours: float,
    ) -> float:
        z = (
            math.log(
                horizon_hours
            )
            - raw_prediction
        ) / self.scale

        return 0.5 * (
            1.0
            + math.erf(
                z
                / math.sqrt(2.0)
            )
        )

    def _build_feature_row(
        self,
        sensor_id: int,
        at: datetime,
    ) -> list[Any]:
        if (
            sensor_id
            not in self.metadata
        ):
            raise KeyError(
                f"Unknown sensor_id="
                f"{sensor_id}"
            )

        self.store.advance(
            sensor_id,
            at,
        )

        state = self.store.get_state(
            sensor_id
        )

        metadata = self.metadata[
            sensor_id
        ]

        weather_by_offset: dict[
            int,
            Mapping[
                str,
                float | int | None,
            ],
        ] = {}

        base_hour = at.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        for offset_seconds in (
            self.compiled
            .weather_offsets_seconds
        ):
            target_hour = (
                base_hour
                + timedelta(
                    seconds=offset_seconds
                )
            )

            weather_by_offset[
                offset_seconds
            ] = self.weather_cache.get(
                target_hour
            )

        row: list[Any] = [
            None
        ] * len(
            self.compiled.specs
        )

        for idx, spec in enumerate(
            self.compiled.specs
        ):
            if spec.kind == "category":
                if spec.seconds == 0:
                    latest = (
                        state.latest_event
                    )

                    value = int(
                        latest is not None
                        and latest.timestamp
                        == at
                        and latest.category
                        == spec.name
                    )
                else:
                    value = (
                        state
                        .category_counts[
                            spec.seconds
                        ]
                        .get(
                            spec.name or "",
                            0,
                        )
                    )

            elif spec.kind == "numeric":
                if spec.seconds == 0:
                    latest = (
                        state.latest_event
                    )

                    value = (
                        latest.numeric
                        if latest
                        is not None
                        and latest.timestamp
                        == at
                        else None
                    )
                else:
                    count = (
                        state
                        .numeric_count[
                            spec.seconds
                        ]
                    )

                    value = (
                        state
                        .numeric_sum[
                            spec.seconds
                        ]
                        / count
                        if count
                        else None
                    )

            elif spec.kind == "weather":
                weather = (
                    weather_by_offset[
                        spec.seconds
                    ]
                )

                if (
                    spec.name
                    not in weather
                ):
                    raise RuntimeError(
                        "Weather service "
                        "response has no "
                        f"{spec.name!r} "
                        f"for "
                        f"{base_hour + timedelta(seconds=spec.seconds)}"
                    )

                value = weather[
                    spec.name
                ]

            elif spec.kind == "day_of_week":
                value = (
                    at.isoweekday()
                )

            else:
                if (
                    spec.name
                    not in metadata
                ):
                    raise KeyError(
                        f"Sensor "
                        f"{sensor_id} "
                        "metadata has no "
                        f"feature "
                        f"{spec.name!r}"
                    )

                value = metadata[
                    spec.name
                ]

            if (
                idx
                in self.cat_feature_indices
            ):
                row[idx] = (
                    "null"
                    if value is None
                    else str(value)
                )
            else:
                row[idx] = (
                    float("nan")
                    if value is None
                    else value
                )

        return row

    def _explain(
        self,
        row: list[Any],
        raw_prediction: float,
        horizon_hours: float,
    ) -> dict[str, Any]:
        started = (
            time.perf_counter()
        )

        pool = Pool(
            data=[row],
            cat_features=list(
                self.cat_feature_indices
            ),
            feature_names=list(
                self.feature_names
            ),
        )

        shap_matrix = (
            self.model
            .get_feature_importance(
                data=pool,
                type="ShapValues",
            )
        )

        shap_values = (
            shap_matrix[0][:-1]
        )

        base_value = float(
            shap_matrix[0][-1]
        )

        values = {
            feature: float(
                shap_value
            )
            for feature, shap_value
            in zip(
                self.feature_names,
                shap_values,
            )
        }

        top_indices = sorted(
            range(
                len(shap_values)
            ),
            key=lambda idx: abs(
                float(
                    shap_values[idx]
                )
            ),
            reverse=True,
        )[:10]

        top = []

        for idx in top_indices:
            shap_value = float(
                shap_values[idx]
            )

            probability_without = (
                self._probability_from_raw(
                    raw_prediction
                    - shap_value,
                    horizon_hours,
                )
            )

            probability_with = (
                self._probability_from_raw(
                    raw_prediction,
                    horizon_hours,
                )
            )

            top.append(
                {
                    "feature": (
                        self.feature_names[
                            idx
                        ]
                    ),
                    "feature_value": (
                        _json_safe(
                            row[idx]
                        )
                    ),
                    "shap_raw": (
                        shap_value
                    ),
                    "risk_direction": (
                        "increases_probability"
                        if shap_value < 0
                        else (
                            "decreases_probability"
                            if shap_value > 0
                            else "neutral"
                        )
                    ),
                    "probability_delta_from_component": (
                        probability_with
                        - probability_without
                    ),
                }
            )

        return {
            "base_value": base_value,
            "values": values,
            "top": top,
            "reconstructed_raw_prediction": (
                base_value
                + sum(
                    float(value)
                    for value
                    in shap_values
                )
            ),
            "calculation_ms": (
                time.perf_counter()
                - started
            ) * 1000,
        }

    def warmup(
        self,
        logs: Iterable[
            Mapping[str, Any]
        ],
    ) -> int:
        n = 0

        with self.lock:
            for payload in logs:
                sensor_id = int(
                    payload[ID_COL]
                )

                if (
                    sensor_id
                    not in self.metadata
                ):
                    continue

                timestamp = (
                    parse_datetime(
                        payload[
                            DATE_COL
                        ],
                        payload[
                            TIME_COL
                        ],
                    )
                )

                self.store.update(
                    sensor_id=(
                        sensor_id
                    ),
                    timestamp=(
                        timestamp
                    ),
                    value=payload.get(
                        VALUE_COL
                    ),
                )

                n += 1

        self.logger.info(
            "warmup_complete "
            "rows=%d sensors=%d",
            n,
            len(self.store.states),
        )

        return n

    def predict(
        self,
        payload: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        started = (
            time.perf_counter()
        )

        sensor_id = int(
            payload[ID_COL]
        )

        timestamp = (
            parse_datetime(
                payload[DATE_COL],
                payload[TIME_COL],
            )
        )

        shap_mode = (
            normalize_shap_mode(
                payload.get(
                    "add_shap",
                    "none",
                )
            )
        )

        horizon_hours = (
            normalize_horizon_hours(
                payload[
                    HORIZON_HOURS_FIELD
                ]
            )
        )

        with self.lock:
            self.store.update(
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=payload.get(
                    VALUE_COL
                ),
            )

            row = (
                self._build_feature_row(
                    sensor_id=sensor_id,
                    at=timestamp,
                )
            )

            predict_started = (
                time.perf_counter()
            )

            raw_prediction = float(
                self.model.predict(
                    [row]
                )[0]
            )

            predict_ms = (
                time.perf_counter()
                - predict_started
            ) * 1000

            probability = (
                self._probability_from_raw(
                    raw_prediction,
                    horizon_hours,
                )
            )

            calculate_shap = (
                should_calculate_shap(
                    mode=shap_mode,
                    probability=(
                        probability
                    ),
                    threshold=(
                        self.config
                        .shap_auto_threshold
                    ),
                )
            )

            result: dict[
                str,
                Any,
            ] = {
                ID_COL: sensor_id,
                "datetime": (
                    timestamp
                    .isoformat()
                ),
                HORIZON_HOURS_FIELD: (
                    horizon_hours
                ),
                "probability": (
                    probability
                ),
                "raw_prediction": (
                    raw_prediction
                ),
                "history_complete": (
                    self.store
                    .history_complete(
                        sensor_id,
                        timestamp,
                    )
                ),
                "shap_mode": (
                    shap_mode
                ),
                "shap_calculated": (
                    calculate_shap
                ),
                "predict_ms": (
                    predict_ms
                ),
            }

            if calculate_shap:
                result["shap"] = (
                    self._explain(
                        row=row,
                        raw_prediction=(
                            raw_prediction
                        ),
                        horizon_hours=(
                            horizon_hours
                        ),
                    )
                )

            self.n_predictions += 1

            if (
                self.config
                .state_save_every_n
                > 0
                and self.n_predictions
                % self.config
                .state_save_every_n
                == 0
            ):
                self.save_state()

        result["total_ms"] = (
            time.perf_counter()
            - started
        ) * 1000

        self.logger.info(
            "prediction "
            "sensor_id=%s "
            "datetime=%s "
            "horizon_hours=%.6f "
            "probability=%.6f "
            "shap=%s "
            "total_ms=%.3f",
            sensor_id,
            timestamp.isoformat(),
            horizon_hours,
            probability,
            calculate_shap,
            result["total_ms"],
        )

        return result

    def save_state(
        self,
    ) -> None:
        self.config.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot = {
            "feature_names": (
                self.feature_names
            ),
            "tree_count": (
                self.model.tree_count_
            ),
            "states": (
                self.store.states
            ),
        }

        tmp_path = (
            self.config.state_path
            .with_suffix(
                self.config
                .state_path
                .suffix
                + ".tmp"
            )
        )

        with tmp_path.open(
            "wb"
        ) as file:
            pickle.dump(
                snapshot,
                file,
                protocol=(
                    pickle
                    .HIGHEST_PROTOCOL
                ),
            )

        tmp_path.replace(
            self.config.state_path
        )

        self.logger.info(
            "state_saved "
            "path=%s sensors=%d",
            self.config.state_path,
            len(self.store.states),
        )

    def load_state_if_exists(
        self,
    ) -> bool:
        if not (
            self.config
            .state_path
            .exists()
        ):
            return False

        with (
            self.config
            .state_path
            .open("rb")
        ) as file:
            snapshot = (
                pickle.load(file)
            )

        if (
            tuple(
                snapshot[
                    "feature_names"
                ]
            )
            != self.feature_names
        ):
            raise RuntimeError(
                "Saved state belongs "
                "to another feature set"
            )

        if (
            int(
                snapshot[
                    "tree_count"
                ]
            )
            != int(
                self.model.tree_count_
            )
        ):
            raise RuntimeError(
                "Saved state belongs "
                "to another model"
            )

        self.store.states = (
            snapshot["states"]
        )

        self.logger.info(
            "state_loaded "
            "path=%s sensors=%d",
            self.config.state_path,
            len(self.store.states),
        )

        return True

    def close(
        self,
    ) -> None:
        self.weather_client.close()


def create_app(
    service: PredictionService,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        yield
        service.save_state()
        service.close()

    app = FastAPI(
        title=(
            "Sensor SurvivalAft "
            "Prediction Service"
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[
        str,
        Any,
    ]:
        return {
            "status": "ok",
            "features": len(
                service.feature_names
            ),
            "trees": (
                service.model.tree_count_
            ),
            "metadata_sensors": len(
                service.metadata
            ),
            "state_sensors": len(
                service.store.states
            ),
            "predictions": (
                service.n_predictions
            ),
            "shap_auto_threshold": (
                service.config
                .shap_auto_threshold
            ),
            "weather_service_url": (
                service.config
                .weather_service_url
            ),
        }

    @app.post("/predict")
    def predict(
        payload: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        required = {
            ID_COL,
            DATE_COL,
            TIME_COL,
            VALUE_COL,
            HORIZON_HOURS_FIELD,
        }

        missing = (
            required
            - set(payload)
        )

        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Missing fields: "
                    f"{sorted(missing)}"
                ),
            )

        try:
            return service.predict(
                payload
            )
        except PredictionRequestError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    @app.post("/save-state")
    def save_state() -> dict[
        str,
        Any,
    ]:
        service.save_state()

        return {
            "saved": True,
            "path": str(
                service.config
                .state_path
            ),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config.yml"
        ),
    )

    args = parser.parse_args()

    config = (
        ServiceConfig.from_yaml(
            args.config
        )
    )

    service = PredictionService(
        config=config
    )

    app = create_app(
        service
    )

    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        workers=1,
    )


if __name__ == "__main__":
    main()
