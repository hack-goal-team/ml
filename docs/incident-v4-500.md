# Incident v4 500

The second inference container uses `data/incident4/model_500.cbm` and
`PREDICTION_KIND=INCIDENT`. It writes FIRE, FLOOD, GAS or INTRUSION with a
30-hour horizon. The equipment container retains `CHANNEL_EVENT` and its
own cursor and model version.

`hours_since_prev_target` is the time since the previous target alarm on the
same channel, excluding the current event. On restart, the runner loads its
checkpoint only if it matches the event cursor. Otherwise it queries alarm
history from Postgres. Before the first production start, verify that the
`alarm_backfill` marker exists and inspect `EXPLAIN ANALYZE` for the history
query on the stand. The backfill changes `events.is_alarm`; the source journal
alarm flag for intrusion is no longer separately available in Postgres, so
historical intrusion targets cannot exactly reproduce training labels.

At most eight incident SHAP explanations start per tick, and no new one starts
after 500 ms of accumulated SHAP time. Predictions beyond this limit still
write with `shap = null`; `shap_skipped` appears in the runner log. For the
interpretation report, the Backend hides `тип_датчика` from the displayed top
causes and fills the list with the next strongest feature. The stored incident
SHAP retains eleven candidates so that ten remain available for display; the
model input and raw SHAP audit still contain `тип_датчика`. The report must
disclose this display filter.
