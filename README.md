# Moscow Hack 2026 — Prediction Service V2

Production-сервис для online-прогноза критического события по логам датчиков на произвольном положительном горизонте в часах.

Модель: `CatBoostRegressor` с `SurvivalAft`.

Сервис принимает новые логи по одному, вместе с каждым запросом получает `horizon_hours`, обновляет rolling-фичи датчика, запрашивает необходимые погодные данные во внутреннем weather-service и возвращает `probability` события до указанного горизонта. При необходимости дополнительно возвращается SHAP.

## Файлы

```text
app.py
demo.py
test_app.py
config.yml
```

### `app.py`

Production-код:

- загрузка CatBoost-модели;
- загрузка и кеширование metadata;
- хранение online rolling-state;
- формирование фичей;
- HTTP-запросы во внутренний weather-service;
- кеширование погоды;
- prediction;
- SurvivalAft → `probability` для переданного `horizon_hours`;
- optional SHAP;
- FastAPI;
- сохранение и восстановление state;
- runtime-логирование.

В production-файле нет mock-логики.

### `demo.py`

Демонстрационный запуск:

- читает реальные логи из parquet;
- поднимает mock внутреннего weather-service;
- поднимает prediction API;
- прогревает rolling-state;
- отправляет логи в `/predict` последовательно по одному;
- показывает примеры `add_shap="all"`, `"auto"` и `"none"`;
- выводит latency и RPS.

Mock weather-service существует только здесь и в тестах.

### `test_app.py`

Тесты:

- YAML-конфига;
- режимов SHAP;
- HTTP-контракта weather-service;
- weather cache;
- rolling-window семантики;
- соответствия incremental rolling brute-force расчёту.

### `config.yml`

Все основные production-параметры:

- host и port prediction API;
- путь до модели;
- metadata;
- runtime-файлы;
- SHAP threshold;
- URL внутреннего weather-service;
- weather timeout;
- cache;
- logging.

## Установка

Python 3.11+.

```bash
pip install polars catboost fastapi uvicorn requests pyyaml pytest
```

## Конфигурация

Пример `config.yml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

model:
  path: "survival_optuna/best_model.cbm"

metadata:
  sensors_path: "data/справочник_каналов_датчиков.parquet"
  objects_path: "data/справочник_объектов_диспетчер.parquet"

runtime:
  metadata_cache_path: "runtime/metadata_cache.pkl"
  state_path: "runtime/feature_state.pkl"
  log_path: "runtime/service.log"
  state_save_every_n: 0

shap:
  auto_threshold: 0.50

weather_service:
  url: "http://127.0.0.1:8010/weather"
  timeout_seconds: 5.0
  timezone: "Europe/Moscow"
  cache_max_entries: 256

logging:
  max_bytes: 52428800
  backup_count: 3
```

Для production обычно достаточно заменить:

```yaml
weather_service:
  url: "http://weather-service:8010/weather"
```

на реальный внутренний адрес погодного сервиса.

## Запуск production

Из корня проекта:

```bash
python app.py \
    --config config.yml
```

Если файлы лежат непосредственно в текущей директории:

```bash
python app.py --config config.yml
```

После запуска prediction API слушает:

```text
http://<api.host>:<api.port>
```

При конфиге по умолчанию:

```text
http://0.0.0.0:8000
```

Для локального обращения:

```text
http://127.0.0.1:8000
```

## Основная схема

```text
Platform
   |
   | POST /predict
   | один новый лог
   v
Prediction Service
   |
   | обновление rolling-state
   | metadata
   |
   | POST /weather
   v
Internal Weather Service
   |
   | JSON с погодой
   v
Prediction Service
   |
   | CatBoost SurvivalAft
   | optional SHAP
   v
Platform
```

## Prediction API

### `GET /health`

Проверка состояния сервиса.

Пример:

```bash
curl http://127.0.0.1:8000/health
```

Ответ:

```json
{
  "status": "ok",
  "features": 98,
  "trees": 100,
  "metadata_sensors": 12000,
  "state_sensors": 8500,
  "predictions": 15342,
  "shap_auto_threshold": 0.5,
  "weather_service_url": "http://127.0.0.1:8010/weather"
}
```

### `POST /predict`

Основная ручка.

Один HTTP-запрос соответствует одному новому логу.

Пример:

```json
{
  "ид_канала_данных": 196752,
  "дата": "2026-08-01",
  "время": "23:51:15",
  "значение_датчика": "Неисправен",
  "horizon_hours": 24,
  "add_shap": "auto"
}
```

Обязательные поля:

| Поле | Тип | Описание |
|---|---|---|
| `ид_канала_данных` | `int` или приводимое к `int` | ID канала |
| `дата` | `string` | `YYYY-MM-DD` |
| `время` | `string` | `HH:MM:SS` |
| `значение_датчика` | `string`, `int`, `float`, `null` | Значение текущего события |
| `horizon_hours` | положительное число | Горизонт, до которого считается вероятность события, в часах |

Необязательное поле:

| Поле | Возможные значения | Default |
|---|---|---|
| `add_shap` | `"none"`, `"all"`, `"auto"` | `"none"` |

`horizon_hours` задаётся в каждом запросе и не является фичей CatBoost. Модель возвращает параметр SurvivalAft-распределения, а сервис переводит его в вероятность `P(T <= horizon_hours)`. Поэтому одной и той же моделью можно запросить, например, 6, 12, 24 или 48 часов без переобучения. Единицы горизонта должны совпадать с единицами времени, использованными при обучении; в этом сервисе это часы.

Примеры:

```text
horizon_hours=6   -> вероятность события в ближайшие 6 часов
horizon_hours=24  -> вероятность события в ближайшие 24 часа
horizon_hours=48  -> вероятность события в ближайшие 48 часов
```

#### `add_shap="none"`

SHAP не считается.

Максимально быстрый режим.

#### `add_shap="all"`

SHAP рассчитывается для каждого запроса.

#### `add_shap="auto"`

SHAP рассчитывается только если:

```text
probability > shap.auto_threshold
```

При default:

```text
probability > 0.50
```

Пример ответа без SHAP:

```json
{
  "ид_канала_данных": 196752,
  "datetime": "2026-08-01T23:51:15",
  "horizon_hours": 24.0,
  "probability": 0.0843,
  "raw_prediction": 3.72,
  "history_complete": true,
  "shap_mode": "auto",
  "shap_calculated": false,
  "predict_ms": 0.41,
  "total_ms": 0.58
}
```

Если SHAP был рассчитан, дополнительно появляется:

```json
{
  "shap": {
    "base_value": 5.47,
    "values": {
      "feature_1": 0.12,
      "feature_2": -0.18
    },
    "top": [
      {
        "feature": "feature_2",
        "feature_value": 4,
        "shap_raw": -0.18,
        "risk_direction": "increases_probability",
        "probability_delta_from_component": 0.031
      }
    ],
    "reconstructed_raw_prediction": 5.72,
    "calculation_ms": 58.4
  }
}
```

Возможные `risk_direction`:

```text
increases_probability
decreases_probability
neutral
```

SHAP объясняет решение модели, а не доказывает причинно-следственную связь.

### `POST /save-state`

Принудительно сохраняет rolling-state.

Запрос:

```json
{}
```

Ответ:

```json
{
  "saved": true,
  "path": "runtime/feature_state.pkl"
}
```

При штатном завершении сервиса state также сохраняется автоматически.

## Внутренний weather-service

Prediction-сервис сам делает HTTP-запросы во внутренний погодный сервис.

URL задаётся в:

```yaml
weather_service:
  url: "http://127.0.0.1:8010/weather"
```

Используется:

```text
POST <weather_service.url>
Content-Type: application/json
```

Prediction-сервис отправляет:

```json
{
  "datetime": "2026-08-01T18:00:00",
  "timezone": "Europe/Moscow"
}
```

`datetime` всегда округлён до начала часа.

Например, для входного события:

```text
2026-08-01 14:37:20
```

модель может потребовать:

```text
14:00 current
15:00 +1h
18:00 +4h
22:00 +8h
02:00 +12h
06:00 +16h
10:00 +20h
14:00 +24h
```

Prediction-сервис последовательно запрашивает недостающие часы.

Важно: обозначения `+1h`, `+4h`, ..., `+24h` в этом разделе относятся к погодным фичам, которые зафиксированы в `model.feature_names_`. Они не ограничивают `horizon_hours` запроса: prediction horizon может быть любым положительным числом часов.

Уже полученные часы кешируются в RAM.

### Ожидаемый ответ weather-service

HTTP `2xx` и JSON object:

```json
{
  "temperature_2m (°C)": 12.4,
  "relative_humidity_2m (%)": 71,
  "precipitation_probability (%)": 20,
  "precipitation (mm)": 0.0,
  "rain (mm)": 0.0,
  "snowfall (cm)": 0.0,
  "snow_depth (m)": 0.0,
  "weather_code (wmo code)": 3,
  "cloud_cover (%)": 85,
  "pressure_msl (hPa)": 1012.4
}
```

Ожидаемые ключи:

```text
temperature_2m (°C)
relative_humidity_2m (%)
precipitation_probability (%)
precipitation (mm)
rain (mm)
snowfall (cm)
snow_depth (m)
weather_code (wmo code)
cloud_cover (%)
pressure_msl (hPa)
```

Значения:

```text
float | int | null
```

Если weather-service:

- недоступен;
- не отвечает до `timeout_seconds`;
- возвращает HTTP error;
- возвращает не JSON object;
- не возвращает нужную модели погодную фичу;

prediction завершается ошибкой, а `/predict` возвращает `503`.

## Работа с metadata

При старте сервис читает:

```text
data/справочник_каналов_датчиков.parquet
data/справочник_объектов_диспетчер.parquet
```

Из них строится:

```text
sensor_id -> metadata
```

После первого построения metadata сохраняется в:

```text
runtime/metadata_cache.pkl
```

Если файла ещё нет, это нормально: директория и cache создаются автоматически.

Cache переиспользуется только пока исходные parquet и набор static-фичей не изменились.

## Rolling-state

Сервис хранит incremental-состояние отдельно для каждого датчика:

- события, необходимые rolling-окнам;
- category counts;
- numeric sums;
- numeric counts;
- watermark;
- последнее событие.

Семантика окна соответствует training:

```text
(t - window, t]
```

Левая граница не входит, текущее событие входит.

Логи одного датчика должны поступать в неубывающем порядке времени.

Допустимо:

```text
15:05:00
15:05:00
15:06:10
```

Недопустимо:

```text
15:10:00
15:05:00
```

Для out-of-order события `/predict` возвращает `409`.

## State persistence

Основной state сохраняется в:

```yaml
runtime:
  state_path: "runtime/feature_state.pkl"
```

При следующем старте он автоматически загружается, если:

- совпадает набор model feature names;
- совпадает количество деревьев модели.

Автосохранение во время работы:

```yaml
runtime:
  state_save_every_n: 0
```

`0` означает, что периодическое сохранение отключено.

Например:

```yaml
state_save_every_n: 10000
```

означает сохранение каждые 10 000 predictions.

## HTTP-коды `/predict`

| Код | Значение |
|---:|---|
| `200` | Prediction успешно рассчитан |
| `404` | Неизвестный sensor ID / отсутствующая metadata |
| `409` | Out-of-order событие или другой `ValueError` состояния |
| `422` | Не передано обязательное поле или `horizon_hours` не является положительным конечным числом |
| `503` | Ошибка внутреннего weather-service или другая runtime-зависимость |

## Запуск demo

Demo самостоятельно поднимает:

```text
1. Mock internal weather-service
2. Prediction API
```

После этого читает реальные логи и отправляет их в prediction API по одному.

```bash
python demo.py \
    --config config.yml \
    --logs data/ext-journal-2026.parquet \
    --n 1000 \
    --horizon-hours 24
```

По умолчанию:

```text
logs = data/ext-journal-2026.parquet
n = 1000
horizon_hours = 24.0
```

Первые три prediction демонстрируют:

```text
1 -> add_shap="all"
2 -> add_shap="auto"
3 -> add_shap="none"
```

Остальные выполняются с:

```text
add_shap="none"
```

В конце выводятся:

- ответы первых трёх запросов;
- общее время;
- средняя HTTP latency;
- RPS.

## Тесты

Запускать через `pytest`, а не через обычный `python`.

```bash
python -m pytest test_app.py -q
```

Успешный результат выглядит примерно так:

```text
.......                                                                  [100%]
7 passed in ...
```

Тест HTTP weather-service реально поднимает локальный HTTP server и проверяет, что production-client отправляет:

```json
{
  "datetime": "2026-09-22T14:00:00",
  "timezone": "Europe/Moscow"
}
```

Также проверяется, что второй запрос за тот же час берётся из weather cache и не вызывает второй HTTP-request.

## Логи

Runtime-лог:

```text
runtime/service.log
```

Пример:

```text
prediction sensor_id=196752 datetime=2026-08-01T23:51:15 horizon_hours=24.000000 probability=0.084300 shap=False total_ms=0.580
```

Ротация задаётся:

```yaml
logging:
  max_bytes: 52428800
  backup_count: 3
```

## Production flow одного события

```text
1. Платформа отправляет POST /predict с новым логом и `horizon_hours`.

2. Сервис проверяет sensor ID и timestamp.

3. Текущий лог добавляется в rolling-state.

4. Устаревшие события исключаются из rolling-окон.

5. Из metadata берутся static-фичи.

6. Определяются необходимые погодные часы.

7. Уже закешированные часы берутся из RAM.

8. Для отсутствующих часов выполняется POST во внутренний weather-service.

9. Формируется feature vector в порядке model.feature_names_.

10. CatBoost возвращает SurvivalAft raw prediction; `horizon_hours` в CatBoost как фича не передаётся.

11. Raw prediction переводится в `probability` для переданного `horizon_hours`.

12. По add_shap определяется необходимость SHAP.

13. JSON-ответ возвращается платформе.
```

## Быстрый старт

```bash
pip install polars catboost fastapi uvicorn requests pyyaml pytest

python -m pytest test_app.py -q

python app.py \
    --config config.yml
```

Перед production-запуском должен быть доступен внутренний weather-service по URL из:

```yaml
weather_service.url
```
---
---
---  

# ПОЛУЧЕНИЕ  ПОГОДНЫХ ДАННЫХ  
(модель шлет запросы в этот сервис)  
Для погодных признаков можно использовать **Open-Meteo**. Для исторических прогнозов используется Historical Forecast API:

`https://historical-forecast-api.open-meteo.com/v1/forecast`

Нам нужны почасовые значения для Москвы (`55.7558, 37.6173`) в таймзоне `Europe/Moscow`.

Модель использует следующие параметры погоды:

* температура;
* относительная влажность;
* вероятность осадков;
* количество осадков;
* дождь;
* снег;
* глубина снега;
* weather code;
* облачность;
* давление на уровне моря.

Пример получения погоды за конкретный час сразу в формате, который ожидает модель:

```python
from datetime import datetime

import polars as pl
import requests


WEATHER_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

WEATHER_COLUMNS = {
    "temperature_2m": "temperature_2m (°C)",
    "relative_humidity_2m": "relative_humidity_2m (%)",
    "precipitation_probability": "precipitation_probability (%)",
    "precipitation": "precipitation (mm)",
    "rain": "rain (mm)",
    "snowfall": "snowfall (cm)",
    "snow_depth": "snow_depth (m)",
    "weather_code": "weather_code (wmo code)",
    "cloud_cover": "cloud_cover (%)",
    "pressure_msl": "pressure_msl (hPa)",
}


def get_weather(target_hour: datetime) -> dict:
    target_hour = target_hour.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    date = target_hour.date().isoformat()

    params = {
        "latitude": 55.7558,
        "longitude": 37.6173,
        "start_date": date,
        "end_date": date,
        "hourly": list(WEATHER_COLUMNS),
        "timezone": "Europe/Moscow",
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    df = (
        pl.DataFrame(data["hourly"])
        .with_columns(
            pl.col("time").str.to_datetime()
        )
        .filter(
            pl.col("time") == target_hour
        )
        .rename(WEATHER_COLUMNS)
    )

    if df.height != 1:
        raise RuntimeError(
            f"Weather not found for {target_hour}"
        )

    return (
        df
        .select(WEATHER_COLUMNS.values())
        .row(0, named=True)
    )
```

Например:

```python
weather = get_weather(
    datetime(2026, 1, 2, 14)
)
```

Результат будет иметь вид:

```python
{
    "temperature_2m (°C)": -4.2,
    "relative_humidity_2m (%)": 81,
    "precipitation_probability (%)": 20,
    "precipitation (mm)": 0.0,
    "rain (mm)": 0.0,
    "snowfall (cm)": 0.0,
    "snow_depth (m)": 0.12,
    "weather_code (wmo code)": 3,
    "cloud_cover (%)": 76,
    "pressure_msl (hPa)": 1015.3,
}
```

Именно такой словарь с названиями полей выше ожидает сервис модели для каждого необходимого часа (`current`, `+1h`, `+4h`, `+8h`, `+12h`, `+16h`, `+20h`, `+24h`).

