"""Draft pending labels for PRs a person marked bot-mislabeled.

For each new PR, finds every Auto PR Triage run before the label, reads the
owner categories each assigned from its run log, and records the reviewer
changes people made after the bot first acted. The draft goes to labels/pending/ with an empty
reason; a person fills in reason, trims wrong_owners, and moves it to labels/.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from bench.github import REPOSITORY, fetch_run_log, gh_api
from bench.labels import LABELS_DIR, PENDING_DIR, BotRun, MislabelRecord, ReviewerChange


MISLABEL_LABEL = "bot-mislabeled"
BOT_LOGIN = "github-actions[bot]"
WORKFLOW_FILE = "auto-pr-triage.yml"
# The bot labels and requests reviewers a few minutes after its run starts.
RUN_LOOKBACK = timedelta(minutes=30)
# Bot actions closer together than this belong to the same run.
BOT_ACTION_GAP = timedelta(minutes=10)
REVIEWER_EVENTS = {"review_requested": "requested", "review_request_removed": "removed"}


def parse_time(value: str, /) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_time(value: datetime, /) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_mislabeled_prs() -> list[int]:
    query = f'repo:{REPOSITORY} is:pr label:"{MISLABEL_LABEL}"'
    items = gh_api("search/issues", params={"q": query, "per_page": "100"})["items"]
    return sorted(item["number"] for item in items)


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


def draft_label(pr_number: int, /) -> MislabelRecord:
    pr = gh_api(f"repos/{REPOSITORY}/pulls/{pr_number}")
    timeline = gh_api(f"repos/{REPOSITORY}/issues/{pr_number}/timeline?per_page=100", paginate=True)
    mislabel_event = max(
        (e for e in timeline if e.get("event") == "labeled" and e["label"]["name"] == MISLABEL_LABEL),
        key=lambda e: e["created_at"],
    )
    mislabeled_at = parse_time(mislabel_event["created_at"])
    bot_action_times = group_bot_actions(
        [
            parse_time(e["created_at"])
            for e in timeline
            if (e.get("actor") or {}).get("login") == BOT_LOGIN
            and e.get("event") in ("labeled", "review_requested")
            and parse_time(e["created_at"]) <= mislabeled_at
        ]
    )
    runs = {
        run["id"]: run
        for run in (find_bot_run(pr=pr, before=time) for time in bot_action_times)
        if run is not None
    }
    bot_runs = tuple(
        BotRun(
            run_id=run["id"],
            created_at=run["created_at"],
            head_sha=run["head_sha"],
            additional_owners=read_bot_owners(run_id=run["id"], pr_number=pr_number),
        )
        for run in sorted(runs.values(), key=lambda run: run["created_at"])
    )
    first_bot_action = min(bot_action_times, default=mislabeled_at)
    reviewer_changes = tuple(
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
    )
    return MislabelRecord(
        pr=pr_number,
        head_sha=bot_runs[-1].head_sha if bot_runs else pr["head"]["sha"],
        bot_runs=bot_runs,
        wrong_owners=tuple(sorted({o for run in bot_runs for o in run.additional_owners})),
        reviewer_changes=reviewer_changes,
        labeled_by=mislabel_event["actor"]["login"],
        labeled_at=mislabeled_at.date().isoformat(),
        reason="",
    )


def main() -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for pr_number in find_mislabeled_prs():
        name = f"{pr_number}.json"
        if (LABELS_DIR / name).exists() or (PENDING_DIR / name).exists():
            continue
        record = draft_label(pr_number)
        (PENDING_DIR / name).write_text(json.dumps(record.to_dict(), indent=2) + "\n")
        print(f"drafted labels/pending/{name}: {len(record.bot_runs)} bot runs, wrong_owners {list(record.wrong_owners)}")


if __name__ == "__main__":
    main()
