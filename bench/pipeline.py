"""Fetch the Auto PR Triage pipeline at a pytorch commit and prepare it to run."""

from __future__ import annotations

import shutil
from pathlib import Path

from bench.github import REPOSITORY, gh_api
from bench.cases import REPO_ROOT


PIPELINE_DIR = "scripts/auto_pr_triage"
CONFIG_DIR = ".github/auto-pr-triage"
CONFIG_PATHS = (
    "CODEOWNERS",
    f"{CONFIG_DIR}/extra_ownership_metadata.json",
    f"{CONFIG_DIR}/team_members.json",
)
CACHE_DIR = REPO_ROOT / "cache" / "pipeline"
# Snapshots are taken of PRs the bot already handled, which intake would skip.
HANDLED_CHECK = "    return bool(names & HANDLED_LABELS)\n"
HANDLED_CHECK_DISABLED = "    return False  # pr-triage-bench: evaluate handled PRs\n"


def fetch_pipeline(pytorch_sha: str, /) -> Path:
    """Download the pipeline scripts and trusted config once per commit."""

    root = CACHE_DIR / pytorch_sha
    if (root / ".complete").exists():
        return root
    shutil.rmtree(root, ignore_errors=True)
    entries = gh_api(f"repos/{REPOSITORY}/contents/{PIPELINE_DIR}?ref={pytorch_sha}")
    paths = [f"{PIPELINE_DIR}/{e['name']}" for e in entries if e["type"] == "file"]
    for path in paths + list(CONFIG_PATHS):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            gh_api(f"repos/{REPOSITORY}/contents/{path}?ref={pytorch_sha}", raw=True)
        )
    (root / ".complete").touch()
    return root


def prepare_pipeline(*, pytorch_sha: str, dest: Path) -> Path:
    """Copy the pipeline at pytorch_sha to dest and admit already-handled PRs."""

    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(fetch_pipeline(pytorch_sha), dest)
    intake = dest / PIPELINE_DIR / "assess_intake.py"
    source = intake.read_text()
    if source.count(HANDLED_CHECK) != 1:
        raise RuntimeError("assess_intake.py changed; update HANDLED_CHECK")
    intake.write_text(source.replace(HANDLED_CHECK, HANDLED_CHECK_DISABLED))
    return dest
