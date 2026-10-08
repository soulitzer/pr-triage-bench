"""Score a run against the regression cases, per owner category.

For each case and expected owner, a run passes when it assigns an owner the
case expects assigned, or leaves out one it expects not assigned. Per
category: recall on cases expecting it, false positive rate on cases expecting
it absent, and precision on this suite (correct assignments over all
assignments). Precision depends on how many cases of each kind the suite has.
Runs whose worker output failed validation are counted separately.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bench.cases import Case, canonical_owner, load_cases, load_owner_aliases


@dataclass(frozen=True)
class CaseResult:
    """How the runs on one case treated one expected owner."""

    pr: int
    owner: str
    expectation: str
    judged_runs: int
    assigned: int
    assigned_with_discarded: int
    failed_runs: int
    rep_owners: tuple[tuple[str, ...] | None, ...]

    @property
    def passed(self) -> int:
        return self.assigned if self.expectation == "assign" else self.judged_runs - self.assigned


@dataclass(frozen=True)
class CategorySummary:
    owner: str
    results: tuple[CaseResult, ...]

    def _sum(self, *, expectation: str, field: str) -> int:
        return sum(getattr(r, field) for r in self.results if r.expectation == expectation)

    @property
    def recall(self) -> tuple[int, int]:
        return self._sum(expectation="assign", field="assigned"), self._sum(
            expectation="assign", field="judged_runs"
        )

    @property
    def false_positive_rate(self) -> tuple[int, int]:
        return self._sum(expectation="not_assign", field="assigned"), self._sum(
            expectation="not_assign", field="judged_runs"
        )

    @property
    def precision(self) -> tuple[int, int]:
        correct = self.recall[0]
        return correct, correct + self.false_positive_rate[0]

    @property
    def failed_runs(self) -> int:
        return sum(r.failed_runs for r in self.results)


def score_run(run_dir: Path, /, *, cases: list[Case]) -> dict[str, CategorySummary]:
    aliases = load_owner_aliases()
    runs_by_pr = defaultdict(list)
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        result = json.loads(line)
        runs_by_pr[result["pr"]].append(result)
    results_by_owner = defaultdict(list)
    for case in cases:
        runs = sorted(runs_by_pr.get(case.pr, []), key=lambda r: r["rep"])
        if not runs:
            continue
        succeeded = [r for r in runs if r["llm_run_status"] == "succeeded"]
        accepted = [{canonical_owner(o, aliases=aliases) for o in r["additional_owners"]} for r in succeeded]
        proposed = [
            owners | {canonical_owner(o, aliases=aliases) for o in r["discarded_owners"]}
            for owners, r in zip(accepted, succeeded)
        ]
        for owner_id, expectation in sorted(case.expected.items()):
            owner = canonical_owner(owner_id, aliases=aliases)
            results_by_owner[owner].append(
                CaseResult(
                    pr=case.pr,
                    owner=owner,
                    expectation=expectation,
                    judged_runs=len(succeeded),
                    assigned=sum(owner in owners for owners in accepted),
                    assigned_with_discarded=sum(owner in owners for owners in proposed),
                    failed_runs=len(runs) - len(succeeded),
                    rep_owners=tuple(
                        tuple(r["additional_owners"]) if r["llm_run_status"] == "succeeded" else None
                        for r in runs
                    ),
                )
            )
    return {
        owner: CategorySummary(owner=owner, results=tuple(results))
        for owner, results in sorted(results_by_owner.items())
    }


def format_ratio(ratio: tuple[int, int], /) -> str:
    numerator, denominator = ratio
    rate = f"{numerator / denominator:.0%}" if denominator else "n/a"
    return f"{rate} ({numerator}/{denominator})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir
    meta = json.loads((run_dir / "meta.json").read_text())
    print({key: meta[key] for key in ("pytorch_sha", "config", "model", "effort", "reps")})
    summaries = score_run(run_dir, cases=load_cases())
    for summary in summaries.values():
        print(
            f"\n{summary.owner}: precision on suite {format_ratio(summary.precision)}, "
            f"recall {format_ratio(summary.recall)}, "
            f"false positive rate {format_ratio(summary.false_positive_rate)}, "
            f"failed runs {summary.failed_runs}"
        )
        for r in summary.results:
            print(f"  #{r.pr}\texpect {r.expectation}\tpassed {r.passed}/{r.judged_runs}\tfailed {r.failed_runs}")


if __name__ == "__main__":
    main()
