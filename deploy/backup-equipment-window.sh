#!/usr/bin/env bash
# Снимок входов модели оборудования перед сменой образа.
# Окно ядра восстанавливается из events; runtime хранит курсор.
set -euo pipefail
umask 077

sha=${SHA:?SHA is required}
[[ $sha =~ ^[[:xdigit:]]{40}$ ]] || { echo 'invalid SHA' >&2; exit 1; }
backup_mount=${BACKUP_MOUNT:-/mnt/backups}
mountpoint -q "$backup_mount" || { echo "$backup_mount is not mounted" >&2; exit 1; }
cd "${REMOTE_DIR:-$HOME/backend}"

services=$(docker compose config --services </dev/null)
if ! grep -qx inference <<<"$services"; then
  echo 'equipment inference is not configured; window cannot be preserved' >&2
  exit 1
fi
container=$(docker compose ps -q inference </dev/null)
[[ -n $container ]] || { echo 'equipment inference container is missing' >&2; exit 1; }
[[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || {
  echo 'equipment inference is not running' >&2
  exit 1
}
old_health=$(docker inspect -f '{{.State.Health.Status}}' "$container")
schema_recovery=0
old_logs=$(docker logs --tail 100 "$container" 2>&1)
if grep -Fq 'column "journal_is_alarm" does not exist' <<<"$old_logs"; then
  schema_recovery=1
  echo 'old equipment is blocked by missing journal_is_alarm; preserving its window before repair'
elif [[ $old_health != healthy ]]; then
  echo "old equipment is not healthy: $old_health" >&2
  exit 1
fi
image=$(docker image inspect -f '{{.Id}}' inference:current)
[[ -n $image ]] || { echo 'inference:current image is missing' >&2; exit 1; }
volume=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/app/runtime"}}{{.Name}}{{end}}{{end}}' "$container")
[[ -n $volume ]] || { echo 'equipment runtime volume is missing' >&2; exit 1; }

# Backups stay on the mounted backup disk, never on the root filesystem.
root="$backup_mount/inference-windows"
install -d -m 0700 "$root"
available=$(df --output=avail -B1 "$backup_mount" | tail -1)
(( available >= 1073741824 )) || {
  echo "backup disk has only $available bytes free" >&2
  exit 1
}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
name="equipment-${stamp}-${sha}"
[[ ! -e "$root/$name" ]] || { echo "$root/$name already exists" >&2; exit 1; }
tmp=$(mktemp -d "$root/.${name}.XXXXXX")
stopped=0
cleanup() {
  local result=$?
  trap - EXIT
  if (( stopped )); then
  if ! docker compose start inference </dev/null || ! verify_old_running; then
      echo 'CRITICAL: old equipment inference did not recover after backup failure' >&2
      result=1
    fi
  fi
  if (( result != 0 )); then
    rm -rf -- "$tmp"
  fi
  exit "$result"
}
trap cleanup EXIT

wait_healthy() {
  local deadline=$((SECONDS + 2100)) health
  while :; do
    health=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo missing)
    [[ $health == healthy ]] && return 0
    if [[ $health == unhealthy || $health == missing ]] || (( SECONDS >= deadline )); then
      echo "equipment inference health after snapshot: $health" >&2
      return 1
    fi
    sleep 15
  done
}

verify_old_running() {
  if (( schema_recovery )); then
    [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]]
  else
    wait_healthy
  fi
}

# Stop before copying the cursor so it matches a completed batch.
stopped=1
docker compose stop inference </dev/null
snapshot_at=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
docker cp "$container:/app/runtime/." - > "$tmp/runtime.tar" </dev/null
docker cp "$container:/app/data/best_model.cbm" "$tmp/equipment_model.cbm" </dev/null
docker cp "$container:/app/data/feature_encoding.json" "$tmp/feature_encoding.json" </dev/null
docker cp "$container:/app/config.yml" "$tmp/config.yml" </dev/null
tar -tf "$tmp/runtime.tar" > /dev/null
docker compose start inference </dev/null
stopped=0
[[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || {
  echo 'equipment inference failed to restart after snapshot' >&2
  exit 1
}

db=$(docker compose exec -T postgres printenv POSTGRES_DB </dev/null | tr -d '\r')
dbuser=$(docker compose exec -T postgres printenv POSTGRES_USER </dev/null | tr -d '\r')
[[ -n $db && -n $dbuser ]] || { echo 'database name/user missing' >&2; exit 1; }
copy_query() {
  local query=$1 destination=$2
  docker compose exec -T postgres \
    psql -X -q -v ON_ERROR_STOP=1 -U "$dbuser" -d "$db" \
    -c "COPY ($query) TO STDOUT WITH (FORMAT csv, HEADER true)" </dev/null \
    | gzip -1 > "$tmp/$destination"
  gzip -t "$tmp/$destination"
}

# The timestamp is fixed across files. Committed inputs up to this instant
# cover all events processed by the saved cursor.
range="ts >= '${snapshot_at}'::timestamptz - interval '72 hours' AND ts <= '${snapshot_at}'::timestamptz"
copy_query "SELECT id, event_id, channel_id, ts, raw_value, is_alarm FROM events WHERE $range" events.csv.gz
# Future-dated rows could already have been processed by the saved cursor.
copy_query "SELECT id, event_id, channel_id, ts, raw_value, is_alarm FROM events WHERE ts > '${snapshot_at}'::timestamptz" future_events.csv.gz
copy_query 'SELECT channel_id, eng_system_type, sensor_type, system_tag, sensor_name, object_id FROM dim_channels_current' channels.csv.gz
copy_query 'SELECT object_id, hierarchy_level, parent_id, object_kind, dispatcher_name FROM dim_objects_current' objects.csv.gz
verify_old_running

python3 - "$tmp" "$snapshot_at" "$image" "$sha" "$volume" "$old_health" "$schema_recovery" <<'PY'
import csv
import gzip
import json
import sys
from pathlib import Path

directory, snapshot_at, image, sha, volume, old_health, schema_recovery = sys.argv[1:]
root = Path(directory)
counts = {}
for name in ('events', 'future_events', 'channels', 'objects'):
    with gzip.open(root / f'{name}.csv.gz', 'rt', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        counts[name] = sum(1 for _ in reader)
        if not header:
            raise SystemExit(f'{name}: empty header')
if not counts['events'] or not counts['channels']:
    raise SystemExit(f'incomplete snapshot: {counts}')
cursor = root / 'events_cursor.json'
import tarfile
with tarfile.open(root / 'runtime.tar') as archive:
    member = next((m for m in archive.getmembers() if m.name.lstrip('./') == cursor.name), None)
    if member is None:
        raise SystemExit('runtime archive has no events_cursor.json')
    stream = archive.extractfile(member)
    if stream is None:
        raise SystemExit('cursor is not a regular file')
    data = json.load(stream)
    if not isinstance(data['low'], int) or not isinstance(data['seen'], list):
        raise SystemExit('invalid cursor')
manifest = {
    'format': 1,
    'snapshot_at_utc': snapshot_at,
    'window_hours': 72,
    'image_id': image,
    'runtime_volume': volume,
    'old_health': old_health,
    'schema_recovery': schema_recovery == '1',
    'deploy_sha': sha,
    'cursor_low': data['low'],
    'cursor_seen': len(data['seen']),
    'rows': counts,
    'inputs': ['events.csv.gz', 'future_events.csv.gz',
               'channels.csv.gz', 'objects.csv.gz',
               'runtime.tar', 'equipment_model.cbm',
               'feature_encoding.json', 'config.yml'],
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY

(cd "$tmp" && sha256sum events.csv.gz future_events.csv.gz channels.csv.gz objects.csv.gz runtime.tar equipment_model.cbm feature_encoding.json config.yml manifest.json > SHA256SUMS && sha256sum -c SHA256SUMS)
(( $(df --output=avail -B1 "$backup_mount" | tail -1) >= 134217728 )) || {
  echo 'backup disk reserve below 128 MiB' >&2
  exit 1
}
mv -- "$tmp" "$root/$name"
trap - EXIT
echo "equipment window saved: $root/$name"
