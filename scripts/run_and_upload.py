"""Container entrypoint: run llm-eval-forge then upload the report to GCS.

Designed for Cloud Run Jobs. Reads RESULTS_BUCKET from env, runs the eval CLI,
then uploads results/full-evaluation-run.json to gs://$RESULTS_BUCKET/reports/<ts>/.

Application Default Credentials are picked up automatically when running on
Cloud Run — no service account JSON needs to be mounted.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    bucket_name = os.environ.get("RESULTS_BUCKET")
    if not bucket_name:
        print("ERROR: RESULTS_BUCKET env var not set", file=sys.stderr)
        sys.exit(1)

    config_path = os.environ.get("EVAL_CONFIG", "config/default.yaml")

    print(f"[{_ts()}] Starting eval run with config={config_path}", flush=True)
    eval_result = subprocess.run(
        ["llm-eval-forge", "run", "--config", config_path],
        cwd="/app",
    )
    print(f"[{_ts()}] Eval CLI exited with code {eval_result.returncode}", flush=True)

    report_path = Path("/app/results/full-evaluation-run.json")
    if not report_path.exists():
        print(f"ERROR: Report not found at {report_path}", file=sys.stderr)
        sys.exit(2)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    blob_name = f"reports/{timestamp}/full-evaluation-run.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(report_path))

    print(f"[{_ts()}] Uploaded report to gs://{bucket_name}/{blob_name}", flush=True)

    # Propagate the eval CLI's exit code so Cloud Run marks the job correctly
    sys.exit(eval_result.returncode)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
