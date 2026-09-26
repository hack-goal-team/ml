#!/usr/bin/env python3
"""Run one isolated retraining job on a Yandex Compute Cloud VM."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CONFIG = json.loads(Path("/etc/ml-job.json").read_text())
BUCKET = CONFIG["bucket"]
RUN_ID = CONFIG["run_id"]
JOB_PREFIX = f"jobs/{RUN_ID}"
ROOT = Path("/work")
REPO = ROOT / "ml"
TRAINING = REPO / "retraining"
RESULTS = f"results/{RUN_ID}"
METADATA_URL = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
BASE_URL = f"https://storage.yandexcloud.net/{BUCKET}/"
MAX_TRAINING_SECONDS = 68 * 3600

state = {"phase": "starting", "started_at": datetime.now(timezone.utc).isoformat()}
state_lock = threading.Lock()
stop_reporter = threading.Event()


def token() -> str:
    request = urllib.request.Request(METADATA_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)["access_token"]


def request_object(key: str, method: str, data: bytes | None = None):
    headers = {"Authorization": f"Bearer {token()}"}
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(BASE_URL + key, data=data, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=120)


def download(key: str, destination: Path, expected_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with request_object(key, "GET") as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: {key}")
    os.replace(temporary, destination)


def upload(key: str, data: bytes) -> None:
    with request_object(key, "PUT", data) as response:
        response.read()


def upload_file(key: str, path: Path) -> None:
    # Final model and log files fit within the single PUT object limit.
    if path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError(f"Artifact is larger than 1 GiB: {path}")
    upload(key, path.read_bytes())


def metrics() -> dict:
    result = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        memory = {line.split(":")[0]: int(line.split()[1]) for line in lines}
        result["ram_used_gib"] = round((memory["MemTotal"] - memory["MemAvailable"]) / 1048576, 2)
        result["ram_total_gib"] = round(memory["MemTotal"] / 1048576, 2)
        disk = shutil.disk_usage(ROOT)
        result["disk_used_gib"] = round(disk.used / 1073741824, 2)
        result["disk_total_gib"] = round(disk.total / 1073741824, 2)
        log_file = ROOT / "run.log"
        if log_file.exists():
            with log_file.open("rb") as handle:
                handle.seek(max(0, log_file.stat().st_size - 4000))
                result["log_tail"] = handle.read().decode(errors="replace")[-3000:]
    except Exception as error:
        result["monitor_error"] = str(error)
    return result


def report() -> None:
    observed = metrics()
    with state_lock:
        for metric in ("ram_used_gib", "disk_used_gib"):
            if metric in observed:
                peak = f"peak_{metric}"
                state[peak] = max(state.get(peak, 0), observed[metric])
        body = dict(state)
    body.update(observed)
    body["updated_at"] = datetime.now(timezone.utc).isoformat()
    upload(f"{RESULTS}/status.json", json.dumps(body, ensure_ascii=False).encode())


def reporter() -> None:
    while not stop_reporter.is_set():
        try:
            report()
        except Exception as error:
            print(f"Status upload failed: {error}", flush=True)
        stop_reporter.wait(120)


def set_phase(phase: str, **details) -> None:
    with state_lock:
        state.update(phase=phase, **details)
    print(f"{datetime.now(timezone.utc).isoformat()} {phase} {details}", flush=True)
    report()


def run_logged(command: list[str], cwd: Path, log_name: str, timeout: int | None = None) -> None:
    log_path = ROOT / log_name
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Timed out after {timeout} seconds: {' '.join(command)}")
    if code:
        raise RuntimeError(f"Process exited with code {code}: {' '.join(command)}")


def upload_artifacts() -> None:
    run_dir = TRAINING / "artifacts" / "runs" / RUN_ID
    if run_dir.exists():
        for path in run_dir.rglob("*"):
            if not path.is_file() or "data" in path.relative_to(run_dir).parts:
                continue
            key = f"{RESULTS}/artifacts/{path.relative_to(run_dir).as_posix()}"
            try:
                upload_file(key, path)
            except Exception as error:
                print(f"Artifact upload failed for {path}: {error}", flush=True)
    for name in ("install.log", "run.log", "baseline.log", "comparison.log"):
        path = ROOT / name
        if path.exists():
            try:
                upload_file(f"{RESULTS}/{name}", path)
            except Exception as error:
                print(f"Log upload failed for {name}: {error}", flush=True)


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()
    try:
        set_phase("downloading_manifest")
        manifest_path = ROOT / "manifest.json"
        download(f"{JOB_PREFIX}/manifest.json", manifest_path)
        manifest = json.loads(manifest_path.read_text())

        set_phase("downloading_code")
        archive = ROOT / "ml-code.tar.gz"
        download(f"{JOB_PREFIX}/ml-code.tar.gz", archive, manifest["code_sha256"])
        REPO.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(REPO, filter="data")

        set_phase("downloading_data")
        incoming = TRAINING / "data" / "incoming_events"
        incoming.mkdir(parents=True, exist_ok=True)
        (incoming / "журнал_событий_пример.parquet").unlink(missing_ok=True)
        for index, file in enumerate(manifest["files"], 1):
            download(f"{manifest['events_prefix']}/{file['name']}", incoming / file["name"], file["sha256"])
            if index % 5 == 0 or index == len(manifest["files"]):
                set_phase("downloading_data", files_downloaded=index, files_total=len(manifest["files"]))

        set_phase("installing_dependencies")
        venv = ROOT / "venv"
        run_logged([sys.executable, "-m", "venv", str(venv)], ROOT, "install.log")
        run_logged([str(venv / "bin" / "python"), "-m", "pip", "install", "-r", "requirements.txt"],
                   TRAINING, "install.log", timeout=3600)

        set_phase("training", run_id=RUN_ID)
        run_logged([str(venv / "bin" / "python"), "-u", "run_pipeline.py", "--run-id", RUN_ID],
                   TRAINING, "run.log", timeout=MAX_TRAINING_SECONDS)
        set_phase("evaluating_baselines")
        run_logged([
            str(venv / "bin" / "python"), "-u", "baseline_previous_day.py",
            "--run-dir", str(TRAINING / "artifacts" / "runs" / RUN_ID),
        ], TRAINING, "baseline.log", timeout=7200)
        set_phase("comparing_models")
        run_logged([
            str(venv / "bin" / "python"), "-u", "compare_old_model.py",
            "--run-dir", str(TRAINING / "artifacts" / "runs" / RUN_ID),
            "--old-model", str(REPO / "data" / "best_model.cbm"),
        ], TRAINING, "comparison.log", timeout=7200)
        set_phase("completed")
    except Exception as error:
        set_phase("failed", error=str(error), traceback=traceback.format_exc()[-3000:])
        raise
    finally:
        upload_artifacts()
        stop_reporter.set()
        thread.join(timeout=5)
        try:
            report()
        except Exception as error:
            print(f"Final status upload failed: {error}", flush=True)


if __name__ == "__main__":
    main()
