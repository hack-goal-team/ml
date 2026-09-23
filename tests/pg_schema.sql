-- Выжимка схемы бэкенда (hack-goal-team/backend, db/changelog):
-- 001-init-schema (dim_channels, dim_objects, districts, weather),
-- 013-weather-model-fields (6 колонок погоды), 004-inference-role (гранты).
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

CREATE ROLE inference LOGIN;
GRANT USAGE ON SCHEMA public TO inference;
GRANT SELECT ON dim_channels, dim_objects, weather TO inference;
