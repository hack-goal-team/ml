# Бэктест HACK-174

Оффлайн-проверка `data/best_model.cbm` на смены апрель-июнь 2026 по тестовым
каналам сплита, тем же кодом признаков, что и в `inference/`/`app.py`, в
двух вариантах — "прогноз ровно на границе смены" и "то, что стенд реально
показывал по последнему событию".

```bash
HACK174_DATASET_DIR=/path/to/Sources/dataset python -m backtest.run
```

Путь к данным (журналы `ext-journal-*.csv` и справочники) обязателен: флагом
`--dataset-dir` или переменной окружения `HACK174_DATASET_DIR` — здесь
намеренно нет дефолта на локальный путь. Результат пишется в
`docs/backtest/` (переопределяется `--output-dir`); `--weather-csv`
указывает на CSV Open-Meteo (по умолчанию — тот, что уже в репозитории,
`retraining/data/reference/`). Остальные флаги — `--help`.

`docs/backtest/predictions.csv` (~99 тыс. строк) не хранится в git —
генерируется каждым прогоном заново, путь в `.gitignore`. Остальные CSV в
`docs/backtest/` — агрегаты, коммитятся.

Определения (смена, эпизод, горизонт, isAlarm, диспетчер и т.д.) — в
`docs/backtest/backtest.md`.
