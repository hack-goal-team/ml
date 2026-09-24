# Инференс для платформы (HACK-137): цикл inference.runner под ролью
# inference. В образ идёт только нужное циклу: ядро, inference/, модель.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INFERENCE_RUNTIME_DIR=/app/runtime \
    TZ=Europe/Moscow

WORKDIR /app

# Зависимости отдельным слоем: правка кода не пересобирает catboost.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Каталог runtime принадлежит пользователю: новый named volume
# наследует владельца из образа, и курсор пишется без root.
RUN useradd --system --uid 10001 --no-create-home inference \
    && mkdir runtime && chown inference runtime

COPY app.py config.yml ./
COPY inference/ inference/
COPY data/best_model.cbm data/best_model.cbm

USER inference
VOLUME /app/runtime

# start-period покрывает загрузку модели и прогрев окон за 3 дня.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
    CMD ["python", "-m", "inference.health"]

CMD ["python", "-m", "inference.runner"]
