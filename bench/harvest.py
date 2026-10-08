"""Draft pending cases, with snapshots, from what people did around the bot.

Drafts go to cases/pending/<pr>/:

1. bot-mislabeled: PRs a person labeled bot-mislabeled. Every category the
   bot assigned starts as expected not_assign.
2. reviewed: with --reviewed-owner <category>, the most recently updated PRs a
   reviewer on that category's roster reviewed. The category starts as
   expected assign.
3. hand-picked: --add <pr> --expect <category>=<assign|not_assign>.

Each draft snapshots the PR's intake result and changed files, and lists the
bot's runs (with the categories each assigned, read from the run log), people's
reviewer changes after the bot first acted, and roster reviews. A person fills
in tests and reason, fixes expected, and moves the directory to cases/. Drafts
not worth keeping go to cases/skipped/ so they are not drafted again.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from bench.cases import (
    CASES_DIR,
    PENDING_DIR,
    REPO_ROOT,
    SKIPPED_DIR,
    BotRun,
    Case,
    Review,
    ReviewerChange,
    canonical_owner,
    load_owner_aliases,
    write_case,
)
from bench.github import REPOSITORY, fetch_run_log, gh_api
from bench.pipeline import prepare_pipeline
from bench.snapshot import take_snapshot


MISLABEL_LABEL = "bot-mislabeled"
BOT_LOGIN = "github-actions[bot]"
WORKFLOW_FILE = "auto-pr-triage.yml"
TEAM_MEMBERS_PATH = ".github/auto-pr-triage/team_members.json"
SNAPSHOT_PIPELINE_DIR = REPO_ROOT / "cache" / "snapshot-pipeline"
# The bot labels and requests reviewers a few minutes after its run starts.
RUN_LOOKBACK = timedelta(minutes=30)
# Bot actions closer together than this belong to the same run.
BOT_ACTION_GAP = timedelta(minutes=10)
REVIEWER_EVENTS = {"review_requested": "requested", "review_request_removed": "removed"}


@dataclass(frozen=True)
class PrHistory:
    """The bot's runs on a PR and what people changed afterwards."""

    pr: dict[str, Any]
    timeline: list[dict[str, Any]]
    bot_runs: tuple[BotRun, ...]
    reviewer_changes: tuple[ReviewerChange, ...]

    @property
    def assigned_owners(self) -> set[str]:
        return {owner for run in self.bot_runs for owner in run.additional_owners}


@dataclass(frozen=True)
class CaseDraft:
    """What a draft knows before its snapshot is taken."""

    history: PrHistory
    expected: dict[str, str]
    source: str
    labeled_by: str
    labeled_at: str
    reviews: tuple[Review, ...] = ()


def parse_time(value: str, /) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_time(value: datetime, /) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def search_prs(query: str, /) -> list[int]:
    """Return matching PR numbers, most recently updated first."""

    items = gh_api(
        "search/issues",
        params={"q": f"repo:{REPOSITORY} is:pr {query}", "sort": "updated", "per_page": "100"},
    )["items"]
    return [item["number"] for item in items]


def find_bot_run(*, pr: dict[str, Any], before: datetime) -> dict[str, Any] | None:
    """Return the last successful triage run for this PR's branch before a time."""

    window = f"{format_time(before - RUN_LOOKBACK)}..{format_time(before)}"
    runs = gh_api(
        f"repos/{REPOSITORY}/actions/workflows/{WORKFLOW_FILE}/runs",
        params={"status": "success", "per_page": "100", "created": window},
    )["workflow_runs"]
    matching = [
        run
        for run in runs
        if (run.get("head_repository") or {}).get("full_name") == pr["head"]["repo"]["full_name"]
        and run["head_branch"] == pr["head"]["ref"]
    ]
    return max(matching, key=lambda run: run["created_at"], default=None)


def read_bot_owners(*, run_id: int, pr_number: int) -> tuple[str, ...]:
    """Read the accepted additional owner IDs from the run's summary line."""

    match = re.search(
        rf"{re.escape(REPOSITORY)}#{pr_number}: llm_run_status=\w+;.*?additional owners=([^;]*);",
        fetch_run_log(run_id),
    )
    if match is None or match.group(1).strip() == "none":
        return ()
    return tuple(owner.strip() for owner in match.group(1).split(","))


def group_bot_actions(times: list[datetime], /) -> list[datetime]:
    """Collapse the bot's labels and requests into one time per run (its last action)."""

    groups: list[list[datetime]] = []
    for time in sorted(times):
        if groups and time - groups[-1][-1] <= BOT_ACTION_GAP:
            groups[-1].append(time)
        else:
            groups.append([time])
    return [group[-1] for group in groups]


def fetch_history(pr_number: int, /, *, until: datetime) -> PrHistory:
    pr = gh_api(f"repos/{REPOSITORY}/pulls/{pr_number}")
    timeline = gh_api(f"repos/{REPOSITORY}/issues/{pr_number}/timeline?per_page=100", paginate=True)
    bot_action_times = group_bot_actions(
        [
            parse_time(e["created_at"])
            for e in timeline
            if (e.get("actor") or {}).get("login") == BOT_LOGIN
            and e.get("event") in ("labeled", "review_requested")
            and parse_time(e["created_at"]) <= until
        ]
    )
    runs = {
        run["id"]: run
        for run in (find_bot_run(pr=pr, before=time) for time in bot_action_times)
        if run is not None
    }
    first_bot_action = min(bot_action_times, default=until)
    return PrHistory(
        pr=pr,
        timeline=timeline,
        bot_runs=tuple(
            BotRun(
                run_id=run["id"],
                created_at=run["created_at"],
                head_sha=run["head_sha"],
                additional_owners=read_bot_owners(run_id=run["id"], pr_number=pr_number),
            )
            for run in sorted(runs.values(), key=lambda run: run["created_at"])
        ),
        reviewer_changes=tuple(
            ReviewerChange(
                at=e["created_at"],
                action=REVIEWER_EVENTS[e["event"]],
                login=(e.get("requested_reviewer") or {}).get("login")
                or f"team:{(e.get('requested_team') or {}).get('slug')}",
                actor=e["actor"]["login"],
            )
            for e in timeline
            if e.get("event") in REVIEWER_EVENTS
            and e["actor"]["login"] != BOT_LOGIN
            and parse_time(e["created_at"]) > first_bot_action
        ),
    )


def draft_mislabeled(pr_number: int, /) -> CaseDraft:
    timeline = gh_api(f"repos/{REPOSITORY}/issues/{pr_number}/timeline?per_page=100", paginate=True)
    mislabel_event = max(
        (e for e in timeline if e.get("event") == "labeled" and e["label"]["name"] == MISLABEL_LABEL),
        key=lambda e: e["created_at"],
    )
    mislabeled_at = parse_time(mislabel_event["created_at"])
    history = fetch_history(pr_number, until=mislabeled_at)
    aliases = load_owner_aliases()
    return CaseDraft(
        history=history,
        expected={canonical_owner(o, aliases=aliases): "not_assign" for o in sorted(history.assigned_owners)},
        source="bot-mislabeled",
        labeled_by=mislabel_event["actor"]["login"],
        labeled_at=mislabeled_at.date().isoformat(),
    )


def draft_reviewed(pr_number: int, /, *, owner: str, reviewer: str) -> CaseDraft | None:
    reviews = tuple(
        Review(at=r["submitted_at"], reviewer=reviewer, state=r["state"])
        for r in gh_api(f"repos/{REPOSITORY}/pulls/{pr_number}/reviews?per_page=100", paginate=True)
        if r["user"]["login"] == reviewer and r.get("submitted_at")
    )
    if not reviews:
        return None
    return CaseDraft(
        history=fetch_history(pr_number, until=datetime.now(timezone.utc)),
        expected={owner: "assign"},
        source="reviewed",
        labeled_by=reviewer,
        labeled_at=reviews[0].at[:10],
        reviews=reviews,
    )


def draft_hand_picked(pr_number: int, /, *, expected: dict[str, str]) -> CaseDraft:
    return CaseDraft(
        history=fetch_history(pr_number, until=datetime.now(timezone.utc)),
        expected=expected,
        source="hand-picked",
        labeled_by=gh_api("user")["login"],
        labeled_at=datetime.now(timezone.utc).date().isoformat(),
    )


def find_roster() -> dict[str, list[str]]:
    """Return current category -> reviewer logins from pytorch main."""

    aliases = load_owner_aliases()
    members = json.loads(gh_api(f"repos/{REPOSITORY}/contents/{TEAM_MEMBERS_PATH}?ref=main", raw=True))
    return {
        canonical_owner(owner, aliases=aliases): [handle.lstrip("@") for handle in handles]
        for owner, handles in members.items()
    }


def is_known(pr_number: int, /) -> bool:
    return any((d / str(pr_number)).exists() for d in (CASES_DIR, PENDING_DIR, SKIPPED_DIR))


def write_pending(draft: CaseDraft, /, *, pytorch_sha: str) -> None:
    """Snapshot the PR with the pipeline at pytorch_sha and write cases/pending/<pr>/."""

    pr_number = draft.history.pr["number"]
    case_dir = PENDING_DIR / str(pr_number)
    shutil.rmtree(case_dir, ignore_errors=True)
    snapshot = take_snapshot(
        pr_number,
        pipeline_root=SNAPSHOT_PIPELINE_DIR / pytorch_sha,
        pytorch_sha=pytorch_sha,
        case_dir=case_dir,
    )
    moved = [run for run in draft.history.bot_runs if run.head_sha != snapshot.head_sha]
    if moved:
        print(f"#{pr_number}: head moved since bot run {moved[-1].run_id}; snapshot is of the current head")
    case = Case(
        pr=pr_number,
        title=draft.history.pr["title"],
        expected=draft.expected,
        tests="",
        source=draft.source,
        labeled_by=draft.labeled_by,
        labeled_at=draft.labeled_at,
        reason="",
        snapshot=snapshot,
        bot_runs=draft.history.bot_runs,
        reviewer_changes=draft.history.reviewer_changes,
        reviews=draft.reviews,
    )
    write_case(case, case_dir=case_dir)
    print(f"drafted cases/pending/{pr_number}: {draft.source}, expected {draft.expected}")


def parse_expectation(value: str, /) -> tuple[str, str]:
    owner, _, expectation = value.partition("=")
    if expectation not in ("assign", "not_assign"):
        raise argparse.ArgumentTypeError("use <category>=assign or <category>=not_assign")
    return owner, expectation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-owner", action="append", default=[], help="category (repeatable)")
    parser.add_argument("--reviewed-limit", type=int, default=10, help="max reviewed drafts per reviewer")
    parser.add_argument("--add", type=int, action="append", default=[], help="hand-picked PR (repeatable)")
    parser.add_argument("--expect", type=parse_expectation, action="append", default=[])
    parser.add_argument("--skip-mislabeled", action="store_true", help="do not draft bot-mislabeled PRs")
    args = parser.parse_args()
    if args.add and not args.expect:
        parser.error("--add needs at least one --expect <category>=<assign|not_assign>")
    pytorch_sha = gh_api(f"repos/{REPOSITORY}/commits/main")["sha"]
    prepare_pipeline(pytorch_sha=pytorch_sha, config_dir=None, dest=SNAPSHOT_PIPELINE_DIR / pytorch_sha)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for pr_number in args.add:
        write_pending(draft_hand_picked(pr_number, expected=dict(args.expect)), pytorch_sha=pytorch_sha)
    mislabeled = [] if args.skip_mislabeled else search_prs(f'label:"{MISLABEL_LABEL}"')
    for pr_number in mislabeled:
        if not is_known(pr_number):
            write_pending(draft_mislabeled(pr_number), pytorch_sha=pytorch_sha)
    roster = find_roster()
    for owner in args.reviewed_owner:
        for reviewer in roster[owner]:
            drafted = 0
            for pr_number in search_prs(f"reviewed-by:{reviewer}"):
                if drafted == args.reviewed_limit:
                    break
                if pr_number in mislabeled or is_known(pr_number):
                    continue
                draft = draft_reviewed(pr_number, owner=owner, reviewer=reviewer)
                if draft is not None:
                    write_pending(draft, pytorch_sha=pytorch_sha)
                    drafted += 1


if __name__ == "__main__":
    main()
