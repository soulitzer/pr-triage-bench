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

from bench.cases import REPO_ROOT, Case, canonical_owner, load_cases, load_owner_aliases
from bench.score import CaseResult, CategorySummary, format_ratio, score_run


RESULTS_DIR = REPO_ROOT / "results"
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
PUBLISHED_FILES = ("results.jsonl", "meta.json")
# (label, CategorySummary field, higher is better), one column each per run.
METRICS = (
    ("precision", "precision", True),
    ("recall", "recall", True),
    ("false pos.", "false_positive_rate", False),
    ("failed", "failure_rate", False),
)
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


def render_run_legend(runs: list[PublishedRun], /) -> str:
    rows = "".join(
        f"<tr><th>{escape(run.name)}</th>"
        f'<td><a href="{PYTORCH_URL}/tree/{run.meta["pytorch_sha"]}/scripts/auto_pr_triage">'
        f'pytorch@{run.meta["pytorch_sha"][:10]}</a></td>'
        f"<td>{render_config_links(run.meta)}</td>"
        f'<td>{escape(run.meta["model"])} ({escape(run.meta["effort"])}), {run.meta["reps"]} reps</td>'
        f'<td>{escape(run.meta["published_at"])}</td></tr>'
        for run in runs
    )
    return (
        "<table><thead><tr><th>run</th><th>pipeline</th><th>config files used</th>"
        f"<th>worker</th><th>published</th></tr></thead><tbody>{rows}</tbody></table>"
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


def render_grid_cell(ratio: tuple[int, int] | None, /, *, higher_is_better: bool) -> str:
    if ratio is None:
        return '<span class="cell na">-</span>'
    numerator, denominator = ratio
    css = ratio_class(ratio, higher_is_better=higher_is_better)
    rate = f"{numerator / denominator:.0%}" if denominator else "n/a"
    return f'<span class="cell {css}">{rate}<span class="sub"> {numerator}/{denominator}</span></span>'


def render_category(owner: str, /, *, cases: list[Case], runs: list[PublishedRun]) -> str:
    """One expandable row: the category's scores per run, then its samples."""

    owner_cases = sorted(
        (case for case in cases if owner in case.expected),
        key=lambda case: (case.expected[owner], case.pr),
    )
    cells = "".join(
        render_grid_cell(
            getattr(run.summaries[owner], field) if owner in run.summaries else None,
            higher_is_better=higher_is_better,
        )
        for run in runs
        for _, field, higher_is_better in METRICS
    )
    summary = (
        f'<summary class="grid"><span class="cell name">{escape(owner)}</span>'
        f'<span class="cell">{len(owner_cases) or "none yet"}</span>{cells}</summary>'
    )
    if not owner_cases:
        return f'<details class="category empty">{summary}<p class="sub">No samples yet.</p></details>'
    run_headers = "".join(f"<th>{escape(run.name)}</th>" for run in runs)
    case_rows = "".join(render_case_row(case, owner=owner, runs=runs) for case in owner_cases)
    return (
        f'<details class="category">{summary}'
        "<table><thead><tr><th>PR</th><th>expected</th><th>what this sample tests</th>"
        f"{run_headers}<th>reason</th><th></th></tr></thead><tbody>{case_rows}</tbody></table>"
        "</details>"
    )


def render_category_table(*, owners: list[str], cases: list[Case], runs: list[PublishedRun]) -> str:
    run_spans = "".join(
        f'<span class="cell head" style="grid-column: span {len(METRICS)}">{escape(run.name)}</span>' for run in runs
    )
    metric_heads = "".join(f'<span class="cell head">{label}</span>' for run in runs for label, _, _ in METRICS)
    header = (
        f'<div class="grid header"><span class="cell head name">category</span>'
        f'<span class="cell head">samples</span>{run_spans}</div>'
        f'<div class="grid header"><span class="cell"></span><span class="cell"></span>{metric_heads}</div>'
    )
    columns = f"16rem 6rem repeat({len(METRICS) * len(runs)}, 6.5rem)"
    rows = "".join(render_category(owner, cases=cases, runs=runs) for owner in owners)
    return f'<div class="categories" style="--columns: {columns}">{header}{rows}</div>'


def render_page(*, cases: list[Case], runs: list[PublishedRun]) -> str:
    aliases = load_owner_aliases()
    owners = sorted(
        {owner for case in cases for owner in case.expected}
        | {canonical_owner(o, aliases=aliases) for run in runs for o in run.meta.get("categories", ())}
    )
    categories = render_category_table(owners=owners, cases=cases, runs=runs)
    legend = render_run_legend(runs)
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
  .categories {{ margin: 1rem 0 2rem; }}
  .grid {{ display: grid; grid-template-columns: var(--columns); border-bottom: 1px solid #d0d7de; }}
  .grid .cell {{ padding: 0.4rem 0.6rem; }}
  .grid .head {{ font-weight: 600; background: #f6f8fa; }}
  .grid .name {{ font-weight: 600; }}
  details.category > summary {{ cursor: pointer; list-style: none; }}
  details.category > summary:hover {{ background: #f6f8fa; }}
  details.category > summary .name::before {{ content: "\\25B8  "; color: #656d76; }}
  details.category[open] > summary .name::before {{ content: "\\25BE  "; }}
  details.category > table, details.category > p {{ margin: 0.5rem 0 1.5rem 1.5rem; }}
  span.cell.good {{ background: #dafbe1; }}
  span.cell.mixed {{ background: #fff8c5; }}
  span.cell.bad {{ background: #ffebe9; }}
  span.cell.na {{ color: #656d76; }}
  td details summary {{ cursor: pointer; }}
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
rather than reading it as a production rate. Click a category to see its
samples. <a href="{REPO_URL}">Source and labeling process</a>.</p>
{categories}
<h2>Runs</h2>
{legend}
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
