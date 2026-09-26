#!/usr/bin/env python3
"""Stage a versioned training job in a private Yandex Object Storage bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[2]
REQUIRED = {"ид_события", "ид_канала_данных", "дата", "время", "значение_датчика", "тревожное"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def include(entry: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(entry.name).parts
    if parts and parts[0] == ".git":
        return None
    if "__pycache__" in parts or ".DS_Store" in parts or ".terraform" in parts:
        return None
    if any(part in {".env", "terraform.tfstate", "terraform.tfstate.backup"}
           or part.endswith((".tfvars", ".key", ".pem")) for part in parts):
        return None
    if parts[:2] == ("retraining", "artifacts"):
        return None
    if parts[:3] == ("retraining", "data", "incoming_events"):
        return None
    return entry


def upload(yc: str, path: Path, bucket: str, key: str) -> None:
    subprocess.run([yc, "storage", "s3", "cp", str(path), f"s3://{bucket}/{key}"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--yc", default="yc", help="Path to authenticated Yandex Cloud CLI")
    parser.add_argument("--reuse-events-prefix", help="Existing S3 prefix with matching files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,50}", args.run_id):
        parser.error("run-id must contain 3-51 lowercase letters, digits or hyphens")
    files = sorted(args.events_dir.glob("*.parquet"))
    if not files:
        parser.error("No Parquet event files found")
    if any(path.name == "журнал_событий_пример.parquet" for path in files):
        parser.error("Do not include the example journal in a full run")

    prefix = f"jobs/{args.run_id}"
    events_prefix = args.reuse_events_prefix or f"{prefix}/events"
    with tempfile.TemporaryDirectory(prefix="ml-training-job-") as temporary:
        scratch = Path(temporary)
        archive = scratch / "ml-code.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(REPO.iterdir()):
                tar.add(path, arcname=path.name, filter=include)
        entries = []
        for path in files:
            scan = pl.scan_parquet(path)
            missing = REQUIRED - set(scan.collect_schema().names())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            entries.append({
                "name": path.name,
                "rows": scan.select(pl.len()).collect().item(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
        manifest = {
            "run_id": args.run_id,
            "events_prefix": events_prefix,
            "code_sha256": digest(archive),
            "total_rows": sum(entry["rows"] for entry in entries),
            "files": entries,
        }
        manifest_path = scratch / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.dry_run:
            if not args.reuse_events_prefix:
                for path in files:
                    upload(args.yc, path, args.bucket, f"{events_prefix}/{path.name}")
            upload(args.yc, archive, args.bucket, f"{prefix}/ml-code.tar.gz")
            upload(args.yc, Path(__file__).with_name("runner.py"), args.bucket, f"{prefix}/runner.py")
            upload(args.yc, manifest_path, args.bucket, f"{prefix}/manifest.json")
        print(json.dumps({
            "run_id": args.run_id,
            "bucket": args.bucket,
            "manifest_key": f"{prefix}/manifest.json",
            "events_prefix": events_prefix,
            "total_rows": manifest["total_rows"],
            "files": len(files),
            "uploaded": not args.dry_run,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
