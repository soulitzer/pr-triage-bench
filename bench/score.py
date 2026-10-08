"""Score a run: per owner category, how often known-wrong assignments come back.

A run "repeats" a mistake when it assigns an owner category that a confirmed
label marks wrong for that PR. Runs whose worker output failed validation are
counted separately, not as repeats or non-repeats.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bench.labels import canonical_owner, load_confirmed_labels, load_owner_aliases


@dataclass(frozen=True)
class MistakeScore:
    """Repeat counts for one wrong owner category, on one PR or summed over PRs."""

    owner: str
    labeled_prs: int
    judged_runs: int
    repeats: int
    repeats_with_discarded: int
    failed_runs: int

    @property
    def repeat_rate(self) -> float | None:
        return self.repeats / self.judged_runs if self.judged_runs else None


@dataclass(frozen=True)
class RunScore:
    by_pr: dict[int, tuple[MistakeScore, ...]]

    @property
    def by_owner(self) -> dict[str, MistakeScore]:
        scores = defaultdict(list)
        for pr_scores in self.by_pr.values():
            for score in pr_scores:
                scores[score.owner].append(score)
        return {
            owner: MistakeScore(
                owner=owner,
                labeled_prs=len(owner_scores),
                judged_runs=sum(s.judged_runs for s in owner_scores),
                repeats=sum(s.repeats for s in owner_scores),
                repeats_with_discarded=sum(s.repeats_with_discarded for s in owner_scores),
                failed_runs=sum(s.failed_runs for s in owner_scores),
            )
            for owner, owner_scores in sorted(scores.items())
        }


def score_run(run_dir: Path, /) -> RunScore:
    aliases = load_owner_aliases()
    runs_by_pr = defaultdict(list)
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        result = json.loads(line)
        runs_by_pr[result["pr"]].append(result)
    by_pr = {}
    for label in load_confirmed_labels():
        succeeded = [r for r in runs_by_pr.get(label.pr, []) if r["llm_run_status"] == "succeeded"]
        accepted = [{canonical_owner(o, aliases=aliases) for o in r["additional_owners"]} for r in succeeded]
        proposed = [
            owners | {canonical_owner(o, aliases=aliases) for o in r["discarded_owners"]}
            for owners, r in zip(accepted, succeeded)
        ]
        wrong_owners = sorted({canonical_owner(o, aliases=aliases) for o in label.wrong_owners})
        by_pr[label.pr] = tuple(
            MistakeScore(
                owner=owner,
                labeled_prs=1,
                judged_runs=len(succeeded),
                repeats=sum(owner in owners for owners in accepted),
                repeats_with_discarded=sum(owner in owners for owners in proposed),
                failed_runs=len(runs_by_pr.get(label.pr, [])) - len(succeeded),
            )
            for owner in wrong_owners
        )
    return RunScore(by_pr=by_pr)


def format_rate(score: MistakeScore, /) -> str:
    rate = "n/a" if score.repeat_rate is None else f"{score.repeat_rate:.0%}"
    return f"{rate} ({score.repeats}/{score.judged_runs})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir
    run_score = score_run(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    print({key: meta[key] for key in ("pytorch_sha", "config", "model", "effort", "reps")})
    print("\ncategory\tlabeled PRs\trepeat rate\tincl. discarded\tfailed runs")
    for score in run_score.by_owner.values():
        print(
            f"{score.owner}\t{score.labeled_prs}\t{format_rate(score)}"
            f"\t{score.repeats_with_discarded}/{score.judged_runs}\t{score.failed_runs}"
        )
    print("\nper PR:")
    for pr, scores in run_score.by_pr.items():
        for score in scores:
            print(f"#{pr}\t{score.owner}\t{format_rate(score)}\tfailed {score.failed_runs}")


if __name__ == "__main__":
    main()
