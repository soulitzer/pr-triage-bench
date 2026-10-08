"""Load and validate human-confirmed mislabel records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "labels"
PENDING_DIR = LABELS_DIR / "pending"
OWNER_ALIASES_PATH = LABELS_DIR / "owner_aliases.json"
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
class MislabelRecord:
    """One PR where the bot assigned owner categories a person judged wrong."""

    pr: int
    head_sha: str
    bot_runs: tuple[BotRun, ...]
    wrong_owners: tuple[str, ...]
    reviewer_changes: tuple[ReviewerChange, ...]
    labeled_by: str
    labeled_at: str
    reason: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MislabelRecord:
        return cls(
            pr=int(value["pr"]),
            head_sha=value["head_sha"],
            bot_runs=tuple(BotRun.from_dict(run) for run in value["bot_runs"]),
            wrong_owners=tuple(value["wrong_owners"]),
            reviewer_changes=tuple(
                ReviewerChange.from_dict(change) for change in value["reviewer_changes"]
            ),
            labeled_by=value["labeled_by"],
            labeled_at=value["labeled_at"],
            reason=value["reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr": self.pr,
            "head_sha": self.head_sha,
            "bot_runs": [run.to_dict() for run in self.bot_runs],
            "wrong_owners": list(self.wrong_owners),
            "reviewer_changes": [change.to_dict() for change in self.reviewer_changes],
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at,
            "reason": self.reason,
        }


def load_owner_aliases() -> dict[str, str]:
    """Map retired owner IDs to their current names."""

    return json.loads(OWNER_ALIASES_PATH.read_text())


def canonical_owner(owner_id: str, /, *, aliases: dict[str, str]) -> str:
    return aliases.get(owner_id, owner_id)


def load_confirmed_labels() -> list[MislabelRecord]:
    """Load labels/<pr>.json, rejecting records a person has not finished."""

    records = []
    for path in sorted(LABELS_DIR.glob("*.json")):
        if path == OWNER_ALIASES_PATH:
            continue
        record = MislabelRecord.from_dict(json.loads(path.read_text()))
        if path.stem != str(record.pr):
            raise ValueError(f"{path.name} holds the label for PR {record.pr}")
        if not record.wrong_owners:
            raise ValueError(f"{path.name}: wrong_owners is empty")
        if not record.reason.strip():
            raise ValueError(f"{path.name}: reason is empty")
        records.append(record)
    return records
