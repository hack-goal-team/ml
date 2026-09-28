#!/usr/bin/env python3
"""Small private HTTP adapter for starting and inspecting cloud training jobs."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,50}\Z")


def handler(config: dict, secret: str):
    state_dir = Path(config["state_dir"]).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    active: dict[str, subprocess.Popen] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code: int, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            if hmac.compare_digest(supplied, f"Bearer {secret}"):
                return True
            self.respond(401, {"error": "unauthorized"})
            return False

        def do_POST(self) -> None:
            if not self.authorized():
                return
            if self.path != "/jobs":
                self.respond(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 1024:
                    raise ValueError("Invalid body size")
                run_id = json.loads(self.rfile.read(size))["runId"]
                if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                    raise ValueError("Invalid runId")
            except (ValueError, KeyError, json.JSONDecodeError):
                self.respond(400, {"error": "Invalid job request"})
                return
            with lock:
                if any(process.poll() is None for process in active.values()):
                    self.respond(409, {"error": "Another job is being provisioned"})
                    return
                if any(not path.with_suffix(".destroyed").exists()
                       for path in state_dir.glob("*.tfstate")):
                    self.respond(409, {"error": "A cloud VM still needs to be destroyed"})
                    return
                if (state_dir / f"{run_id}.log").exists():
                    self.respond(409, {"error": "runId already exists"})
                    return
                command = [sys.executable, str(HERE / "job.py"), "launch", "--run-id", run_id,
                           "--settings", config["settings"], "--state-dir", str(state_dir),
                           "--events-dir", config["events_dir"]]
                if config.get("reuse_events_prefix"):
                    command += ["--reuse-events-prefix", config["reuse_events_prefix"]]
                if config.get("yc"):
                    command += ["--yc", config["yc"]]
                if config.get("tofu"):
                    command += ["--tofu", config["tofu"]]
                with (state_dir / f"{run_id}.log").open("x") as log:
                    active[run_id] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            self.respond(202, {"runId": run_id, "phase": "provisioning"})

        def do_GET(self) -> None:
            if not self.authorized():
                return
            match = re.fullmatch(r"/jobs/([a-z0-9][a-z0-9-]{2,50})", self.path)
            if not match:
                self.respond(404, {"error": "not found"})
                return
            run_id = match.group(1)
            if not (state_dir / f"{run_id}.log").exists():
                self.respond(404, {"error": "unknown runId"})
                return
            command = [sys.executable, str(HERE / "job.py"), "status", "--run-id", run_id,
                       "--settings", config["settings"], "--state-dir", str(state_dir)]
            if config.get("yc"):
                command += ["--yc", config["yc"]]
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.respond(200, json.loads(result.stdout))
                return
            process = active.get(run_id)
            if process is None:
                self.respond(200, {"runId": run_id, "phase": "status_unavailable",
                                   "error": "No cloud status after orchestrator restart; inspect provisioning log"})
            elif process.poll() is None:
                self.respond(200, {"runId": run_id, "phase": "provisioning"})
            else:
                self.respond(200, {"runId": run_id, "phase": "failed",
                                   "error": "Cloud job did not publish status; inspect provisioning log"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    secret = os.environ.get("ML_TRAINING_API_TOKEN", "")
    if len(secret) < 32:
        parser.error("ML_TRAINING_API_TOKEN must contain at least 32 characters")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for required in ("settings", "state_dir", "events_dir"):
        if required not in config:
            parser.error(f"Missing config key: {required}")
    ThreadingHTTPServer((args.host, args.port), handler(config, secret)).serve_forever()


if __name__ == "__main__":
    main()
