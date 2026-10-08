"""Publish runs' sanitized results and regenerate the dashboard page.

Copies results.jsonl and meta.json from runs/<name>/ into results/<name>/
(committed), then rebuilds docs/index.html from every published run and the
current cases. Only owner IDs, counts, and case metadata are published.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from bench.cases import REPO_ROOT, Case, load_cases
from bench.score import CaseResult, CategorySummary, format_ratio, score_run


RESULTS_DIR = REPO_ROOT / "results"
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
PUBLISHED_FILES = ("results.jsonl", "meta.json")
PYTORCH_URL = "https://github.com/pytorch/pytorch"
REPO_URL = "https://github.com/soulitzer/pr-triage-bench"
# Config files a run reads from pytorch, by repo path, unless overridden.
PYTORCH_CONFIG_PATHS = {
    "worker.md": "scripts/auto_pr_triage/worker.md",
    "CODEOWNERS": "CODEOWNERS",
    "extra_ownership_metadata.json": ".github/auto-pr-triage/extra_ownership_metadata.json",
    "team_members.json": ".github/auto-pr-triage/team_members.json",
}


@dataclass(frozen=True)
class PublishedRun:
    name: str
    meta: dict
    summaries: dict[str, CategorySummary]

    def result(self, *, owner: str, pr: int) -> CaseResult | None:
        summary = self.summaries.get(owner)
        return next((r for r in summary.results if r.pr == pr), None) if summary else None


def publish_run(run_dir: Path, /) -> None:
    dest = RESULTS_DIR / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_FILES:
        shutil.copyfile(run_dir / name, dest / name)
    meta = json.loads((dest / "meta.json").read_text())
    meta.setdefault("published_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_published_runs(*, cases: list[Case]) -> list[PublishedRun]:
    runs = [
        PublishedRun(
            name=run_dir.name,
            meta=json.loads((run_dir / "meta.json").read_text()),
            summaries=score_run(run_dir, cases=cases),
        )
        for run_dir in sorted(RESULTS_DIR.iterdir())
        if (run_dir / "results.jsonl").exists()
    ]
    return sorted(runs, key=lambda run: run.meta["published_at"])


def ratio_class(ratio: tuple[int, int], /, *, higher_is_better: bool) -> str:
    numerator, denominator = ratio
    if not denominator:
        return "na"
    best, worst = (denominator, 0) if higher_is_better else (0, denominator)
    return "good" if numerator == best else ("bad" if numerator == worst else "mixed")


def render_ratio_cell(ratio: tuple[int, int] | None, /, *, higher_is_better: bool) -> str:
    if ratio is None:
        return '<td class="na">not run</td>'
    return f'<td class="{ratio_class(ratio, higher_is_better=higher_is_better)}">{escape(format_ratio(ratio))}</td>'


def render_config_links(meta: dict, /) -> str:
    """Link every config file the run used to the exact commit it came from."""

    links = []
    for name, path in PYTORCH_CONFIG_PATHS.items():
        if name in meta.get("config_files", ()):
            config_dir = Path(meta["config"]).name
            url = f"{REPO_URL}/blob/{meta['bench_sha']}/configs/{config_dir}/{name}"
            note = " (override, uncommitted changes)" if meta.get("config_dirty") else " (override)"
        else:
            url = f"{PYTORCH_URL}/blob/{meta['pytorch_sha']}/{path}"
            note = ""
        links.append(f'<a href="{url}">{escape(name)}</a>{note}')
    return "<br>".join(links)


def render_run_header(run: PublishedRun, /) -> str:
    sha = run.meta["pytorch_sha"]
    return (
        f'<th>{escape(run.name)}<br><span class="sub">'
        f'pipeline: <a href="{PYTORCH_URL}/tree/{sha}/scripts/auto_pr_triage">pytorch@{sha[:10]}</a><br>'
        f"{render_config_links(run.meta)}<br>"
        f'{escape(run.meta["model"])} ({escape(run.meta["effort"])}), {run.meta["reps"]} reps'
        f'<br>published {escape(run.meta["published_at"])}</span></th>'
    )


def render_rep_owners(result: CaseResult, /) -> str:
    return "; ".join(
        f"rep {i}: " + ("failed validation" if owners is None else (", ".join(owners) or "none"))
        for i, owners in enumerate(result.rep_owners, start=1)
    )


def render_case_details(case: Case, /, *, owner: str, runs: list[PublishedRun]) -> str:
    rep_lines = "".join(
        f"<li>{escape(run.name)}: {escape(render_rep_owners(result))}</li>"
        for run in runs
        if (result := run.result(owner=owner, pr=case.pr)) is not None
    )
    bot_lines = "".join(
        f"<li>run {run.run_id} ({escape(run.created_at[:10])}): "
        f"{escape(', '.join(run.additional_owners) or 'no extra owners')}</li>"
        for run in case.bot_runs
    ) or "<li>no Auto PR Triage runs</li>"
    people_lines = "".join(
        f"<li>{escape(c.at[:10])}: {escape(c.actor)} {escape(c.action)} {escape(c.login)}</li>"
        for c in case.reviewer_changes
    ) + "".join(
        f"<li>{escape(r.at[:10])}: {escape(r.reviewer)} reviewed ({escape(r.state.lower())})</li>"
        for r in case.reviews
    )
    return (
        "<details><summary>details</summary>"
        f"<b>Benchmark runs</b><ul>{rep_lines}</ul>"
        f"<b>Bot runs on the PR</b><ul>{bot_lines}</ul>"
        f"<b>People</b><ul>{people_lines or '<li>none recorded</li>'}</ul>"
        '<span class="sub">snapshot of PR head '
        f'<a href="{PYTORCH_URL}/pull/{case.pr}/commits/{case.snapshot.head_sha}">{case.snapshot.head_sha[:10]}</a>, '
        f"taken {escape(case.snapshot.taken_at[:10])} with the pipeline at "
        f'<a href="{PYTORCH_URL}/tree/{case.snapshot.pytorch_sha}/scripts/auto_pr_triage">'
        f"pytorch@{case.snapshot.pytorch_sha[:10]}</a></span>"
        "</details>"
    )


def render_case_row(case: Case, /, *, owner: str, runs: list[PublishedRun]) -> str:
    expectation = case.expected[owner]
    cells = []
    for run in runs:
        result = run.result(owner=owner, pr=case.pr)
        if result is None:
            cells.append('<td class="na">not run</td>')
            continue
        failed = f'<br><span class="sub">{result.failed_runs} failed</span>' if result.failed_runs else ""
        css = ratio_class((result.passed, result.judged_runs), higher_is_better=True)
        cells.append(f'<td class="{css}">pass {result.passed}/{result.judged_runs}{failed}</td>')
    return (
        f'<tr><td><a href="{PYTORCH_URL}/pull/{case.pr}">#{case.pr}</a><br>'
        f'<span class="sub">{escape(case.title)}</span></td>'
        f"<td>{escape(expectation.replace('_', ' '))}</td>"
        f"<td>{escape(case.tests)}</td>"
        + "".join(cells)
        + f'<td class="reason">{escape(case.reason)}<br><span class="sub">{escape(case.source)}, '
        f"labeled by {escape(case.labeled_by)} on {escape(case.labeled_at)}</span></td>"
        f"<td>{render_case_details(case, owner=owner, runs=runs)}</td></tr>"
    )


def render_category(owner: str, /, *, cases: list[Case], runs: list[PublishedRun]) -> str:
    headers = "".join(render_run_header(run) for run in runs)
    metric_rows = "".join(
        f"<tr><th>{label}</th>"
        + "".join(
            render_ratio_cell(
                getattr(run.summaries[owner], field) if owner in run.summaries else None,
                higher_is_better=higher_is_better,
            )
            for run in runs
        )
        + "</tr>"
        for label, field, higher_is_better in (
            ("precision on this suite", "precision", True),
            ("recall (assign cases)", "recall", True),
            ("false positive rate (not-assign cases)", "false_positive_rate", False),
        )
    )
    owner_cases = sorted(
        (case for case in cases if owner in case.expected),
        key=lambda case: (case.expected[owner], case.pr),
    )
    case_rows = "".join(render_case_row(case, owner=owner, runs=runs) for case in owner_cases)
    return (
        f"<h2>{escape(owner)}</h2>"
        f"<table><thead><tr><th>metric</th>{headers}</tr></thead><tbody>{metric_rows}</tbody></table>"
        f"<details><summary>{len(owner_cases)} samples</summary>"
        "<table><thead><tr><th>PR</th><th>expected</th><th>what this sample tests</th>"
        f"{headers}<th>reason</th><th></th></tr></thead><tbody>{case_rows}</tbody></table>"
        "</details>"
    )


def render_page(*, cases: list[Case], runs: list[PublishedRun]) -> str:
    owners = sorted({owner for case in cases for owner in case.expected})
    categories = "".join(render_category(owner, cases=cases, runs=runs) for owner in owners)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pr-triage-bench</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; max-width: 1400px; }}
  table {{ border-collapse: collapse; margin: 0.75rem 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.7rem; text-align: left; vertical-align: top; }}
  thead th {{ background: #f6f8fa; }}
  td.good {{ background: #dafbe1; }}
  td.mixed {{ background: #fff8c5; }}
  td.bad {{ background: #ffebe9; }}
  td.na {{ color: #656d76; }}
  td.reason {{ max-width: 26rem; }}
  details {{ margin-bottom: 1.5rem; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  td details {{ margin: 0; }}
  td details summary {{ font-weight: normal; color: #0969da; }}
  .sub {{ color: #656d76; font-size: 0.85em; font-weight: normal; }}
</style>
</head>
<body>
<h1>pr-triage-bench</h1>
<p>Regression suite for PyTorch's Auto PR Triage owner routing. Each sample is a
PR whose inputs were snapshotted, with the owner categories a person says the
bot should or should not assign. Every config run replays the snapshots through
the production pipeline, several times per sample. A run passes a sample when it
assigns the categories expected and leaves out the ones expected absent; runs
that fail validation are reported separately. <b>Precision on this suite</b>
depends on how many samples of each kind the suite has, so compare configs
rather than reading it as a production rate. Expand a category to see its
samples. <a href="{REPO_URL}">Source and labeling process</a>.</p>
{categories}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="*", help="runs to publish before rebuilding")
    for run_dir in parser.parse_args().run_dirs:
        publish_run(run_dir)
    cases = load_cases()
    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PAGE_PATH.write_text(render_page(cases=cases, runs=load_published_runs(cases=cases)))
    print(f"wrote {PAGE_PATH}")


if __name__ == "__main__":
    main()
