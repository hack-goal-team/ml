#!/usr/bin/env python3
"""Stage, provision, inspect and remove an isolated cloud training job."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TERRAFORM = HERE / "terraform"
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,50}\Z")


def command(args: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, env=env)
    return result.stdout.strip()


def settings(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"folder_id", "subnet_id", "security_group_id", "service_account_id", "bucket_name"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing cloud settings: {', '.join(sorted(missing))}")
    return config


def tofu(args: argparse.Namespace, action: str, config: dict) -> None:
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"{args.run_id}.tfstate"
    env = dict(os.environ)
    env["YC_TOKEN"] = command([args.yc, "iam", "create-token"])
    env["TF_DATA_DIR"] = str(state_dir / f"{args.run_id}.terraform")
    env.setdefault("TF_CLI_CONFIG_FILE", str(HERE / "tofurc"))
    base = [args.tofu, f"-chdir={TERRAFORM}"]
    command(base + ["init", "-backend=false", "-input=false"], env=env)
    variables = dict(config, run_id=args.run_id,
                     instance_name=f"ml-training-{args.run_id}")
    allowed = {"folder_id", "zone", "subnet_id", "security_group_id", "service_account_id",
               "bucket_name", "run_id", "instance_name", "cores", "memory_gib", "work_disk_gib"}
    flags = [f"-var={name}={value}" for name, value in variables.items() if name in allowed]
    subprocess.run(base + [action, "-input=false", "-auto-approve", f"-state={state_file}", *flags],
                   check=True, env=env)


def stage(args: argparse.Namespace, config: dict) -> None:
    cmd = [sys.executable, str(HERE / "prepare_job.py"), "--events-dir", str(args.events_dir),
           "--bucket", config["bucket_name"], "--run-id", args.run_id, "--yc", args.yc]
    if args.reuse_events_prefix:
        cmd += ["--reuse-events-prefix", args.reuse_events_prefix]
    subprocess.run(cmd, check=True)


def status(args: argparse.Namespace, config: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="ml-job-status-") as temporary:
        local = Path(temporary) / "status.json"
        command([args.yc, "storage", "s3", "cp",
                 f"s3://{config['bucket_name']}/results/{args.run_id}/status.json", str(local)])
        print(local.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "status", "destroy"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--settings", type=Path, required=True, help="Cloud IDs JSON kept outside Git")
    parser.add_argument("--state-dir", type=Path, required=True, help="Private directory for Terraform state")
    parser.add_argument("--events-dir", type=Path, help="Parquet export; required for launch")
    parser.add_argument("--reuse-events-prefix", help="Existing S3 data prefix for an unchanged export")
    parser.add_argument("--yc", default="yc")
    parser.add_argument("--tofu", default="tofu")
    args = parser.parse_args()
    if not RUN_ID.fullmatch(args.run_id):
        parser.error("Invalid run-id")
    config = settings(args.settings)
    if args.action == "launch":
        if args.events_dir is None:
            parser.error("--events-dir is required for launch")
        if (args.state_dir / f"{args.run_id}.tfstate").exists():
            parser.error("Terraform state already exists for this run-id")
        stage(args, config)
        tofu(args, "apply", config)
    elif args.action == "status":
        status(args, config)
    else:
        if not (args.state_dir / f"{args.run_id}.tfstate").exists():
            parser.error("Terraform state not found; refusing destroy")
        tofu(args, "destroy", config)
        (args.state_dir / f"{args.run_id}.destroyed").write_text("destroyed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
