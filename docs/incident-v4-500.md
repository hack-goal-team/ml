# Incident v4 500

The second inference container uses `data/incident4/model_500.cbm` and
`PREDICTION_KIND=INCIDENT`. It writes FIRE, FLOOD, GAS or INTRUSION with a
30-hour horizon. The equipment container keeps its own cursor and model version. It writes
`EQUIPMENT_FAILURE` only for channels without an incident class: its target is
any alarm, which on a classed channel mixes faults with incidents, so there it
keeps `CHANNEL_EVENT` (hidden by Backend as LEGACY).

`hours_since_prev_target` is the time since the previous target alarm on the
same channel, excluding the current event. `target_seed.json` contains the
last target per channel from 259,283,128 original journal rows in 2019–2020
and 2022–June 2026. The seed is built by `retraining/build_incident_seed.py`;
2021 is excluded as in training. Backend stores a compact checkpoint for
targets between the seed cutoff and the recent replay window. On restart, the
runner loads its cursor checkpoint when it matches; otherwise it merges the
seed and Backend checkpoint, then replays events from `covered_until`.
INTRUSION uses the original `journal_is_alarm` flag preserved by Backend;
a missing flag on a candidate event stops startup. The first rollout restores
this column only for the most recent 72 hours from the SMVU mock journal.
New events retain the original flag in the ingestion handler. Verify the
backfill and replay query on the stand before production.

At most eight incident SHAP explanations start per tick, and no new one starts
after 500 ms of accumulated SHAP time. Predictions beyond this limit still
write with `shap = null`; `shap_skipped` appears in the runner log. For the
interpretation report, the Backend hides `тип_датчика` from the displayed top
causes and fills the list with the next strongest feature. The stored incident
SHAP retains eleven candidates so that ten remain available for display; the
model input and raw SHAP audit still contain `тип_датчика`. The report must
disclose this display filter.
