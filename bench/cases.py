"""Load and validate regression cases.

Each case is a directory cases/<pr>/ holding case.json (the owner categories
the PR should get, and why) and snapshot/ (the PR-side inputs the bot needed:
intake result and changed files), so runs never depend on the live PR.

The cases form one dataset. A case's owners are the complete set of
categories it should get, judged against judged_categories: every other
judged category is expected absent, so every case counts toward every
category's false positive rate. Categories added after a case was labeled are
unjudged for it until someone reviews it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "cases"
PENDING_DIR = CASES_DIR / "pending"
SKIPPED_DIR = CASES_DIR / "skipped"
OWNER_ALIASES_PATH = CASES_DIR / "owner_aliases.json"
CASE_FILE = "case.json"
SNAPSHOT_DIR = "snapshot"
INTAKE_FILE = "intake.json"
FILES_FILE = "files.json"
SOURCES = frozenset({"bot-mislabeled", "reviewed", "hand-picked"})
REVIEWER_ACTIONS = frozenset({"requested", "removed"})


@dataclass(frozen=True)
class ReviewerChange:
    """A reviewer request a person added or removed after the bot ran."""

    at: str
    action: str
    login: str
    actor: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReviewerChange:
        change = cls(
            at=value["at"], action=value["action"], login=value["login"], actor=value["actor"]
        )
        if change.action not in REVIEWER_ACTIONS:
            raise ValueError(f"unknown reviewer change action: {change.action}")
        return change

    def to_dict(self) -> dict[str, str]:
        return {"at": self.at, "action": self.action, "login": self.login, "actor": self.actor}


@dataclass(frozen=True)
class Review:
    """A review a roster reviewer submitted on the PR."""

    at: str
    reviewer: str
    state: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Review:
        return cls(at=value["at"], reviewer=value["reviewer"], state=value["state"])

    def to_dict(self) -> dict[str, str]:
        return {"at": self.at, "reviewer": self.reviewer, "state": self.state}


@dataclass(frozen=True)
class BotRun:
    """One successful Auto PR Triage run on the PR and the owners it added."""

    run_id: int
    created_at: str
    head_sha: str
    additional_owners: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BotRun:
        return cls(
            run_id=int(value["run_id"]),
            created_at=value["created_at"],
            head_sha=value["head_sha"],
            additional_owners=tuple(value["additional_owners"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "head_sha": self.head_sha,
            "additional_owners": list(self.additional_owners),
        }


@dataclass(frozen=True)
class SnapshotInfo:
    """When and from which PR head the snapshot was taken."""

    head_sha: str
    pytorch_sha: str
    taken_at: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SnapshotInfo:
        return cls(
            head_sha=value["head_sha"], pytorch_sha=value["pytorch_sha"], taken_at=value["taken_at"]
        )

    def to_dict(self) -> dict[str, str]:
        return {"head_sha": self.head_sha, "pytorch_sha": self.pytorch_sha, "taken_at": self.taken_at}


@dataclass(frozen=True)
class Case:
    """One PR with the complete set of owner categories it should get."""

    pr: int
    title: str
    owners: tuple[str, ...]
    judged_categories: tuple[str, ...]
    tests: str
    source: str
    labeled_by: str
    labeled_at: str
    reason: str
    snapshot: SnapshotInfo
    bot_runs: tuple[BotRun, ...]
    reviewer_changes: tuple[ReviewerChange, ...]
    reviews: tuple[Review, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Case:
        return cls(
            pr=int(value["pr"]),
            title=value["title"],
            owners=tuple(value["owners"]),
            judged_categories=tuple(value["judged_categories"]),
            tests=value["tests"],
            source=value["source"],
            labeled_by=value["labeled_by"],
            labeled_at=value["labeled_at"],
            reason=value["reason"],
            snapshot=SnapshotInfo.from_dict(value["snapshot"]),
            bot_runs=tuple(BotRun.from_dict(run) for run in value["bot_runs"]),
            reviewer_changes=tuple(
                ReviewerChange.from_dict(change) for change in value["reviewer_changes"]
            ),
            reviews=tuple(Review.from_dict(review) for review in value["reviews"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr": self.pr,
            "title": self.title,
            "owners": list(self.owners),
            "judged_categories": list(self.judged_categories),
            "tests": self.tests,
            "source": self.source,
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at,
            "reason": self.reason,
            "snapshot": self.snapshot.to_dict(),
            "bot_runs": [run.to_dict() for run in self.bot_runs],
            "reviewer_changes": [change.to_dict() for change in self.reviewer_changes],
            "reviews": [review.to_dict() for review in self.reviews],
        }

    @property
    def expected(self) -> dict[str, str]:
        """Each judged category mapped to assign or not_assign."""

        return {
            category: "assign" if category in self.owners else "not_assign"
            for category in self.judged_categories
        }


def write_case(case: Case, /, *, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / CASE_FILE).write_text(json.dumps(case.to_dict(), indent=2) + "\n")


def load_owner_aliases() -> dict[str, str]:
    """Map retired owner IDs to their current names."""

    return json.loads(OWNER_ALIASES_PATH.read_text())


def canonical_owner(owner_id: str, /, *, aliases: dict[str, str]) -> str:
    return aliases.get(owner_id, owner_id)


def load_cases() -> list[Case]:
    """Load cases/<pr>/, rejecting cases a person has not finished."""

    cases = []
    for case_dir in sorted(CASES_DIR.iterdir()):
        if not (case_dir / CASE_FILE).exists():
            continue
        case = Case.from_dict(json.loads((case_dir / CASE_FILE).read_text()))
        problem = None
        if case_dir.name != str(case.pr):
            problem = f"holds the case for PR {case.pr}"
        elif not case.judged_categories or not set(case.owners) <= set(case.judged_categories):
            problem = "owners must be a subset of a non-empty judged_categories"
        elif case.source not in SOURCES:
            problem = f"source must be one of {sorted(SOURCES)}"
        elif not case.tests.strip() or not case.reason.strip():
            problem = "tests and reason must be filled in"
        elif not all((case_dir / SNAPSHOT_DIR / name).exists() for name in (INTAKE_FILE, FILES_FILE)):
            problem = "snapshot is missing"
        if problem:
            raise ValueError(f"cases/{case_dir.name}: {problem}")
        cases.append(case)
    return cases
