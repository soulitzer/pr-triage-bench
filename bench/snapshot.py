"""Snapshot the PR-side inputs the Auto PR Triage worker needs.

Runs the pipeline's intake stage against the live PR and saves its result and
the PR's changed files. Intake facts are recorded as an open, unhandled PR so
the case can be evaluated after the PR is triaged or closed; the worker never
sees these facts, so this does not change what is tested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench.cases import FILES_FILE, INTAKE_FILE, SNAPSHOT_DIR, SnapshotInfo
from bench.github import REPOSITORY, gh_api, gh_token
from bench.pipeline import PIPELINE_DIR


def take_snapshot(
    pr_number: int, /, *, pipeline_root: Path, pytorch_sha: str, case_dir: Path
) -> SnapshotInfo:
    snapshot_dir = case_dir / SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"GH_TOKEN": os.environ.get("GH_TOKEN") or gh_token()}
    intake_run = subprocess.run(
        [sys.executable, str(pipeline_root / PIPELINE_DIR / "assess_intake.py"), str(pr_number),
         "--repository", REPOSITORY, "--workflow-sha", pytorch_sha, "--output-dir", str(snapshot_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    intake_path = snapshot_dir / INTAKE_FILE
    if not intake_path.exists():
        raise RuntimeError(f"intake failed for #{pr_number}: {intake_run.stdout[-500:]}")
    intake = json.loads(intake_path.read_text())
    intake["facts"]["is_open_non_draft_pr_against_main"] = True
    intake["facts"]["is_already_handled"] = False
    intake_path.write_text(json.dumps(intake, indent=2, sort_keys=True) + "\n")
    files = gh_api(f"repos/{REPOSITORY}/pulls/{pr_number}/files?per_page=100", paginate=True)
    (snapshot_dir / FILES_FILE).write_text(json.dumps(files, indent=2) + "\n")
    return SnapshotInfo(
        head_sha=intake["identity"]["head_sha"],
        pytorch_sha=pytorch_sha,
        taken_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
