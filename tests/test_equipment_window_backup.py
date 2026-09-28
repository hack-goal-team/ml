"""The pre-deploy snapshot must keep the existing window and cursor."""
from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/backup-equipment-window.sh"
SHA = "a" * 40


def _mock_tools(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mountpoint = bin_dir / "mountpoint"
    mountpoint.write_text("#!/bin/sh\nexit 0\n")
    mountpoint.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text("""#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$MOCK_LOG"
case "$*" in
  'compose config --services') printf 'postgres\\ninference\\n' ;;
  'compose ps -q inference') echo old-container ;;
  'inspect -f {{.State.Running}} old-container') echo true ;;
  'inspect -f {{range .Mounts}}{{if eq .Destination "/app/runtime"}}{{.Name}}{{end}}{{end}} old-container') echo backend_inference-runtime ;;
  'inspect -f {{.State.Health.Status}} old-container') echo healthy ;;
  'image inspect -f {{.Id}} inference:current') echo sha256:old-image ;;
  'compose stop inference') : ;;
  'compose start inference') [ "${MOCK_FAIL_START:-0}" != 1 ] ;;
  'cp old-container:/app/runtime/. -')
    if [ "${MOCK_FAIL_CP:-0}" = 1 ]; then exit 2; fi
    tar -C "$MOCK_RUNTIME" -cf - . ;;
  'cp old-container:/app/data/best_model.cbm '*|\\
  'cp old-container:/app/data/feature_encoding.json '*|\\
  'cp old-container:/app/config.yml '*)
    cp "$MOCK_MODEL/$(basename "${2#*:}")" "$3" ;;
  'compose exec -T postgres printenv POSTGRES_DB') echo goal ;;
  'compose exec -T postgres printenv POSTGRES_USER') echo goal ;;
  *'FROM events WHERE ts > '*) printf 'id,event_id,channel_id,ts,raw_value,is_alarm\\n' ;;
  *'FROM events WHERE '*) printf 'id,event_id,channel_id,ts,raw_value,is_alarm\\n1,2,3,2026-09-28T00:00:00Z,open,f\\n' ;;
  *'FROM dim_channels_current'*) printf 'channel_id,eng_system_type,sensor_type,system_tag,sensor_name,object_id\\n3,a,b,c,d,4\\n' ;;
  *'FROM dim_objects_current'*) printf 'object_id,hierarchy_level,parent_id,object_kind,dispatcher_name\\n4,a,,b,c\\n' ;;
  *) echo "unexpected docker call: $*" >&2; exit 3 ;;
esac
""")
    docker.chmod(0o755)
    return bin_dir


def _run(tmp_path: Path, fail_copy: bool = False,
         fail_start: bool = False) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = _mock_tools(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "events_cursor.json").write_text('{"low": 7, "seen": [9]}')
    model = tmp_path / "model"
    model.mkdir()
    for name in ("best_model.cbm", "feature_encoding.json", "config.yml"):
        (model / name).write_text(name)
    backup = tmp_path / "backup"
    backup.mkdir()
    log = tmp_path / "calls.log"
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SHA": SHA,
        "BACKUP_MOUNT": str(backup),
        "REMOTE_DIR": str(tmp_path),
        "MOCK_RUNTIME": str(runtime),
        "MOCK_MODEL": str(model),
        "MOCK_LOG": str(log),
        "MOCK_FAIL_CP": "1" if fail_copy else "0",
        "MOCK_FAIL_START": "1" if fail_start else "0",
    }
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True,
                            capture_output=True, check=False)
    return result, backup


def test_snapshot_contains_cursor_window_and_model(tmp_path: Path) -> None:
    result, backup = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    snapshots = list((backup / "inference-windows").glob("equipment-*"))
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.stat().st_mode & 0o777 == 0o700
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["cursor_low"] == 7
    assert manifest["cursor_seen"] == 1
    assert manifest["rows"] == {"events": 1, "future_events": 0,
                                "channels": 1, "objects": 1}
    assert manifest["image_id"] == "sha256:old-image"
    assert manifest["runtime_volume"] == "backend_inference-runtime"
    with tarfile.open(snapshot / "runtime.tar") as archive:
        assert archive.extractfile("./events_cursor.json").read()
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=snapshot, check=True,
                   capture_output=True)
    calls = (tmp_path / "calls.log").read_text()
    assert calls.index("compose stop inference") < calls.index("compose start inference")


def test_failed_cursor_copy_restarts_old_service(tmp_path: Path) -> None:
    result, backup = _run(tmp_path, fail_copy=True)
    assert result.returncode != 0
    calls = (tmp_path / "calls.log").read_text()
    assert "compose start inference" in calls
    assert not list((backup / "inference-windows").glob("equipment-*"))


def test_failed_restart_reports_old_service_unavailable(tmp_path: Path) -> None:
    result, backup = _run(tmp_path, fail_start=True)
    assert result.returncode != 0
    assert "CRITICAL: old equipment inference did not recover" in result.stderr
    assert not list((backup / "inference-windows").glob("equipment-*"))
