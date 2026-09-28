-- Выжимка схемы бэкенда (hack-goal-team/backend, db/changelog):
-- 001-init-schema (dim_channels, dim_objects, districts, weather),
-- 005/012 (view dim_*_current), 013-weather-model-fields (6 колонок
-- погоды), 004-inference-role (гранты). Гранты на view и districts
-- бэкенд добавит в HACK-136.
CREATE TABLE dim_channels (
    channel_id       integer     NOT NULL,
    snapshot_at      timestamptz NOT NULL,
    eng_system_type  text        NOT NULL,
    sensor_type      text        NOT NULL,
    system_tag       text        NOT NULL,
    sensor_name      text,
    object_id        integer,
    PRIMARY KEY (channel_id, snapshot_at)
);

CREATE TABLE dim_objects (
    object_id        integer     NOT NULL,
    snapshot_at      timestamptz NOT NULL,
    hierarchy_level  smallint    NOT NULL,
    parent_id        integer,
    object_kind      text        NOT NULL,
    dispatcher_name  text,
    PRIMARY KEY (object_id, snapshot_at)
);

CREATE TABLE districts (
    district_id  integer PRIMARY KEY,
    name         text    NOT NULL,
    latitude     numeric(8,5) NOT NULL,
    longitude    numeric(8,5) NOT NULL
);

CREATE TABLE weather (
    district_id     integer     NOT NULL REFERENCES districts,
    valid_for       timestamptz NOT NULL,
    fetched_at      timestamptz NOT NULL,
    is_forecast     boolean     NOT NULL,
    temperature_c   numeric(5,2),
    humidity_pct    numeric(5,2),
    precipitation   numeric(6,2),
    pressure_hpa    numeric(7,2),
    precipitation_probability_pct numeric(5,2),
    rain_mm                       numeric(6,2),
    snowfall_cm                   numeric(6,2),
    snow_depth_m                  numeric(5,2),
    weather_code                  smallint,
    cloud_cover_pct               numeric(5,2),
    PRIMARY KEY (district_id, valid_for, is_forecast)
);

CREATE VIEW dim_channels_current AS
SELECT DISTINCT ON (channel_id) *
FROM dim_channels
ORDER BY channel_id, snapshot_at DESC;

CREATE VIEW dim_objects_current AS
SELECT DISTINCT ON (object_id) *
FROM dim_objects
ORDER BY object_id, snapshot_at DESC;

CREATE ROLE inference LOGIN;
GRANT USAGE ON SCHEMA public TO inference;
GRANT SELECT ON dim_channels, dim_objects, weather TO inference;
GRANT SELECT ON dim_channels_current, dim_objects_current, districts
    TO inference;

-- Поток и прогнозы: 001 (events, prediction_log, decisions_on_prediction),
-- 014 (prediction_log.shap), гранты 004. Партиция одна — DEFAULT.
CREATE TABLE events (
    id          bigint      GENERATED ALWAYS AS IDENTITY,
    event_id    bigint      NOT NULL,
    channel_id  integer     NOT NULL,
    ts          timestamptz NOT NULL,
    is_alarm    boolean     NOT NULL,
    raw_value   text        NOT NULL,
    journal_is_alarm boolean,
    PRIMARY KEY (id, ts),
    UNIQUE (event_id, ts)
) PARTITION BY RANGE (ts);
CREATE TABLE events_default PARTITION OF events DEFAULT;

CREATE TABLE alarm_backfill (finished_at timestamptz NOT NULL DEFAULT now());
INSERT INTO alarm_backfill DEFAULT VALUES;
GRANT SELECT ON alarm_backfill TO inference;

CREATE TABLE incident_history_checkpoint (
    id smallint PRIMARY KEY CHECK (id = 1),
    covered_until timestamptz NOT NULL,
    targets jsonb NOT NULL
);
INSERT INTO incident_history_checkpoint VALUES
    (1, '2026-07-01 00:00:00+03', '{}'::jsonb);
GRANT SELECT ON incident_history_checkpoint TO inference;

CREATE TABLE reason_codes (
    code         text PRIMARY KEY,
    description  text NOT NULL
);

CREATE TABLE prediction_log (
    prediction_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    target_kind     text        NOT NULL,
    target_ref      text        NOT NULL,
    incident_type   text        NOT NULL,
    probability     numeric(5,4) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    horizon_until   timestamptz NOT NULL,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    model_version   text        NOT NULL,
    shap            jsonb
);

CREATE TABLE decisions_on_prediction (
    decision_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id  bigint      NOT NULL REFERENCES prediction_log,
    reason_code    text        NOT NULL REFERENCES reason_codes,
    decided_by     text        NOT NULL
);

GRANT SELECT ON events, decisions_on_prediction TO inference;
GRANT SELECT, INSERT, DELETE ON prediction_log TO inference;
