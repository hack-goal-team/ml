# Обучение в Yandex Cloud

`job.py` выполняет три отдельные операции: кладёт снимок кода и Parquet в закрытый Object Storage, создаёт временную VM через Terraform/OpenTofu и читает статус. На VM `bootstrap.py` монтирует отдельный диск, а `runner.py` выполняет пять ноутбуков, сравнивает результат с правилом «тревога вчера» и старой моделью, сохраняет артефакты в `results/<run-id>/` и выключает VM. Terraform state остаётся на машине запуска вне Git. После завершения VM и диск нужно удалить командой `destroy`: выключение VM не удаляет диск.

## Подготовка

1. Установить Python с зависимостью `polars` из `retraining/requirements.txt`, `yc` с доступом к Yandex Cloud и `tofu` (или Terraform). Авторизовать `yc` для загрузки в Object Storage и создания IAM-токена. Сервисный аккаунт VM должен читать `jobs/` и писать `results/` в приватном бакете.
2. Подготовить Parquet с колонками `ид_события`, `ид_канала_данных`, `дата`, `время`, `значение_датчика`, `тревожное`. Для стенда есть `../export_stand_events.py`; пароль SSH передаётся путём к файлу вне Git. Справочники и погода берутся из снимка репозитория.
3. Скопировать `cloud-settings.example.json` в файл вне репозитория и заполнить ID существующих сети, группы безопасности, сервисного аккаунта и бакета. Группа безопасности должна разрешать исходящий HTTPS; входящие порты обучению не нужны. Квота должна вмещать заданные `cores` и `memory_gib`.

## Один запуск

Из корня репозитория:

```bash
python retraining/cloud/job.py launch \
  --run-id full-20260927 \
  --settings /private/path/cloud-settings.json \
  --state-dir /private/path/terraform-state \
  --events-dir /private/path/events-parquet
```

Команда проверяет схему и считает SHA-256 всех файлов, загружает их в `jobs/<run-id>/`, затем выполняет `tofu apply`. При повторном использовании **того же** уже загруженного набора данных можно добавить `--reuse-events-prefix events/r3`; файлы всё равно проверяются по SHA-256 на VM. Для нового набора этот флаг не использовать. Версия кода фиксируется в отдельном архиве каждого запуска.

Статус и уборка:

```bash
python retraining/cloud/job.py status --run-id full-20260927 \
  --settings /private/path/cloud-settings.json --state-dir /private/path/terraform-state
python retraining/cloud/job.py destroy --run-id full-20260927 \
  --settings /private/path/cloud-settings.json --state-dir /private/path/terraform-state
```

Окончание — `phase: completed` в `results/<run-id>/status.json`. `phase: failed` означает, что нужно смотреть `run.log`, `bootstrap-status.json` и выполненные тетрадки в том же префиксе. Перед удалением VM проверьте, что результаты в бакете появились. Модель не заменяется автоматически: сначала сравнить PR-AUC и операционные метрики с правилами «≥1 тревога вчера» и «≥10 тревог вчера» на одном test-срезе.

## API для кнопки

`serve.py` предоставляет `POST /jobs` с `{"runId":"..."}` и `GET /jobs/<run-id>` для Java-бэкенда. Он запускается на отдельном управляющем хосте с теми же `yc`, `tofu`, доступом к каталогу Parquet и приватному state. По умолчанию слушает только `127.0.0.1:8099`; при удалённом доступе нужен закрытый канал до хоста. Установите длинный случайный `ML_TRAINING_API_TOKEN` на этом хосте и тот же токен как `ML_TRAINING_ORCHESTRATOR_TOKEN` в бэкенде. Адрес задаётся `ML_TRAINING_ORCHESTRATOR_URL`. Без этих переменных Java-ручка отвечает 503. Пример файла настроек — `orchestrator.example.json`.

```bash
ML_TRAINING_API_TOKEN='<secret>' python retraining/cloud/serve.py \
  --config /private/path/orchestrator.json
```

Все файлы данных, ключи и Terraform state должны оставаться вне Git. API использует уже подготовленный каталог Parquet на управляющем хосте; обновлять его перед повторным обучением нужно отдельно. Конфигурация не разворачивает управляющий хост и не меняет прод без явной настройки.
