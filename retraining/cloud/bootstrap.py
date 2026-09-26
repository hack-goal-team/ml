#!/usr/bin/env python3
"""Set up the training disk and launch the isolated Yandex Cloud runner."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path


CONFIG = json.loads(Path("/etc/ml-job.json").read_text())
BUCKET = CONFIG["bucket"]
RUN_ID = CONFIG["run_id"]
DISK = Path("/dev/disk/by-id/virtio-ml-data")
WORK = Path("/work")
METADATA_URL = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"


def token() -> str:
    request = urllib.request.Request(METADATA_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)["access_token"]


def s3(key: str, method: str, data: bytes | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token()}"}
    request = urllib.request.Request(
        f"https://storage.yandexcloud.net/{BUCKET}/{key}",
        headers=headers, method=method, data=data,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    # Guard against an OOM kill of both the notebook and this bootstrap process.
    subprocess.run(["shutdown", "-h", "+4320"], check=True)
    print("Waiting for training disk", flush=True)
    for _ in range(120):
        if DISK.exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Training disk not found: {DISK}")

    if subprocess.run(["blkid", str(DISK)], stdout=subprocess.DEVNULL).returncode:
        subprocess.run(["mkfs.ext4", "-F", str(DISK)], check=True)
    WORK.mkdir(exist_ok=True)
    subprocess.run(["mount", str(DISK), str(WORK)], check=True)
    print("Training disk mounted", flush=True)

    subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=1800)
    subprocess.run(["apt-get", "install", "-y", "-qq", "python3-venv"], check=True, timeout=1800)
    runner = WORK / "ml-cloud-runner.py"
    runner.write_bytes(s3(f"jobs/{RUN_ID}/runner.py", "GET"))
    print("Launching ML runner", flush=True)
    subprocess.run([sys.executable, "-u", str(runner)], check=True)


if __name__ == "__main__":
    result = {"ok": True}
    try:
        main()
    except Exception as error:
        result = {"ok": False, "error": str(error), "traceback": traceback.format_exc()[-3000:]}
        print(result["traceback"], flush=True)
    try:
        s3(f"results/{RUN_ID}/bootstrap-status.json", "PUT", json.dumps(result).encode())
    except Exception as error:
        print(f"Could not upload bootstrap status: {error}", flush=True)
    subprocess.run(["shutdown", "-h", "now"], check=False)
