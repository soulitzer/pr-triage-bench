"""Publish runs' sanitized results and regenerate the dashboard page.

Copies results.jsonl and meta.json from runs/<name>/ into results/<name>/
(committed), then rebuilds docs/index.html from every published run and the
current cases. Only owner IDs, counts, and case metadata are published.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from bench.cases import REPO_ROOT, Case, canonical_owner, load_cases, load_owner_aliases
from bench.score import CaseResult, CategorySummary, format_ratio, score_run


RESULTS_DIR = REPO_ROOT / "results"
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
PUBLISHED_FILES = ("results.jsonl", "meta.json")
# The run's ownership metadata, kept so the page can show description changes.
METADATA_FILE = "extra_ownership_metadata.json"
# (label, CategorySummary field, higher is better), one column each per run.
METRICS = (
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
    index: int
    name: str
    meta: dict
    summaries: dict[str, CategorySummary]
    descriptions: dict[str, str]

    @property
    def config_label(self) -> str:
        if not self.meta["config"]:
            return "checked in"
        dirty = ", uncommitted changes" if self.meta.get("config_dirty") else ""
        return f"override: {Path(self.meta['config']).name}{dirty}"

    def result(self, *, owner: str, pr: int) -> CaseResult | None:
        summary = self.summaries.get(owner)
        return next((r for r in summary.results if r.pr == pr), None) if summary else None


def publish_run(run_dir: Path, /) -> None:
    """Copy a run's sanitized files; republishing keeps the first publish time."""

    dest = RESULTS_DIR / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    previous = json.loads((dest / "meta.json").read_text()) if (dest / "meta.json").exists() else {}
    for name in PUBLISHED_FILES:
        shutil.copyfile(run_dir / name, dest / name)
    shutil.copyfile(run_dir / "pipeline" / ".github" / "auto-pr-triage" / METADATA_FILE, dest / METADATA_FILE)
    meta = json.loads((dest / "meta.json").read_text())
    meta["published_at"] = previous.get("published_at") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_descriptions(run_dir: Path, /) -> dict[str, str]:
    path = run_dir / METADATA_FILE
    if not path.exists():
        return {}
    aliases = load_owner_aliases()
    return {
        canonical_owner(owner, aliases=aliases): entry["description"]
        for owner, entry in json.loads(path.read_text()).items()
    }


def load_published_runs(*, cases: list[Case]) -> list[PublishedRun]:
    runs = [
        PublishedRun(
            index=-1,
            name=run_dir.name,
            meta=json.loads((run_dir / "meta.json").read_text()),
            summaries=score_run(run_dir, cases=cases),
            descriptions=load_descriptions(run_dir),
        )
        for run_dir in sorted(RESULTS_DIR.iterdir())
        if (run_dir / "results.jsonl").exists()
    ]
    ordered = sorted(runs, key=lambda run: run.meta["published_at"])
    return [replace(run, index=index) for index, run in enumerate(ordered)]


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
            cells.append(f'<td class="na run-start" data-run="{run.index}">not run</td>')
            continue
        failed = f'<br><span class="sub">{result.failed_runs} failed</span>' if result.failed_runs else ""
        css = ratio_class((result.passed, result.judged_runs), higher_is_better=True)
        cells.append(
            f'<td class="{css} run-start" data-run="{run.index}">pass {result.passed}/{result.judged_runs}{failed}</td>'
        )
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


def render_grid_cell(
    ratio: tuple[int, int] | None, /, *, higher_is_better: bool, starts_run: bool, run_index: int
) -> str:
    edge = " run-start" if starts_run else ""
    if ratio is None:
        return f'<span class="cell na{edge}" data-run="{run_index}">-</span>'
    numerator, denominator = ratio
    css = ratio_class(ratio, higher_is_better=higher_is_better)
    rate = f"{numerator / denominator:.0%}" if denominator else "n/a"
    return (
        f'<span class="cell {css}{edge}" data-run="{run_index}">{rate}'
        f'<span class="sub"> {numerator}/{denominator}</span></span>'
    )


def render_run_heading(run: PublishedRun, /) -> str:
    return (
        f'<span class="slot"></span>{escape(run.name)}<br><span class="sub">config: {escape(run.config_label)}<br>'
        f'pipeline: pytorch@{run.meta["pytorch_sha"][:10]}</span>'
    )


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
            starts_run=index == 0,
            run_index=run.index,
        )
        for run in runs
        for index, (_, field, higher_is_better) in enumerate(METRICS)
    )
    summary = (
        f'<summary class="grid"><span class="cell name">{escape(owner)}</span>'
        f'<span class="cell">{len(owner_cases) or "none yet"}</span>{cells}</summary>'
    )
    if not owner_cases:
        return f'<details class="category empty">{summary}<p class="sub">No samples yet.</p></details>'
    run_headers = "".join(
        f'<th class="run-start" data-run="{run.index}">{render_run_heading(run)}</th>' for run in runs
    )
    case_rows = "".join(render_case_row(case, owner=owner, runs=runs) for case in owner_cases)
    return (
        f'<details class="category">{summary}'
        "<table><thead><tr><th>PR</th><th>expected</th><th>what this sample tests</th>"
        f"{run_headers}<th>reason</th><th></th></tr></thead><tbody>{case_rows}</tbody></table>"
        "</details>"
    )


def render_category_table(*, owners: list[str], cases: list[Case], runs: list[PublishedRun]) -> str:
    run_spans = "".join(
        f'<span class="cell head run-start" data-run="{run.index}" style="grid-column: span {len(METRICS)}">'
        f"{render_run_heading(run)}</span>"
        for run in runs
    )
    metric_heads = "".join(
        f'<span class="cell head{" run-start" if index == 0 else ""}" data-run="{run.index}">{label}</span>'
        for run in runs
        for index, (label, _, _) in enumerate(METRICS)
    )
    header = (
        f'<div class="grid header"><span class="cell head name">category</span>'
        f'<span class="cell head">samples</span>{run_spans}</div>'
        f'<div class="grid header"><span class="cell"></span><span class="cell"></span>{metric_heads}</div>'
    )
    columns = f"16rem 6rem repeat({len(METRICS) * len(runs)}, 6.5rem)"
    rows = "".join(render_category(owner, cases=cases, runs=runs) for owner in owners)
    return f'<div class="categories" style="--columns: {columns}">{header}{rows}</div>'


COMPARE_SCRIPT = r"""
const data = JSON.parse(document.getElementById("bench-data").textContent);
const pickA = document.getElementById("run-a");
const pickB = document.getElementById("run-b");
data.runs.forEach((run, i) => {
  pickA.add(new Option(run.name, i));
  pickB.add(new Option(run.name, i));
});
pickA.value = Math.max(0, data.runs.length - 2);
pickB.value = data.runs.length - 1;

function node(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function row(cells) {
  const tr = node("tr");
  cells.forEach(c => tr.append(c instanceof Node ? c : node("td", c)));
  return tr;
}
function table(headers, rows) {
  const t = node("table");
  const head = node("tr");
  headers.forEach(h => head.append(node("th", h)));
  t.append(head, ...rows);
  return t;
}
function rate(r) { return r && r[1] ? r[0] / r[1] : null; }
function fmt(r) { return !r ? "-" : r[1] ? Math.round(100 * r[0] / r[1]) + "% (" + r[0] + "/" + r[1] + ")" : "n/a"; }
function link(pr) {
  const td = node("td");
  const a = node("a", "#" + pr);
  a.href = "https://github.com/pytorch/pytorch/pull/" + pr;
  td.append(a, node("br"), node("span", data.cases[pr].title, "sub"));
  return td;
}
function same(x, y) { return x === y ? " (same)" : ""; }

// Show only the picked runs' columns, A before B, in the table and sample tables.
function arrangeColumns(order) {
  const shown = order.map(String);
  const groups = new Map();
  document.querySelectorAll("[data-run]").forEach(cell => {
    cell.style.display = shown.includes(cell.dataset.run) ? "" : "none";
    if (!groups.has(cell.parentNode)) groups.set(cell.parentNode, []);
    groups.get(cell.parentNode).push(cell);
  });
  groups.forEach((cells, parent) => {
    const anchor = cells[cells.length - 1].nextSibling;
    shown.forEach(i => cells.filter(c => c.dataset.run === i).forEach(c => parent.insertBefore(c, anchor)));
  });
  document.querySelectorAll("[data-run] .slot").forEach(slot => {
    const i = shown.indexOf(slot.closest("[data-run]").dataset.run);
    slot.textContent = i < 0 ? "" : "ABC"[i] + ": ";
  });
  const grid = document.querySelector(".categories");
  if (grid) grid.style.setProperty("--columns", "16rem 6rem repeat(" + data.metrics.length * shown.length + ", 6.5rem)");
}

// Word-level diff of two texts as [op, text] pieces, op in "=", "-", "+".
function wordDiff(before, after) {
  const x = before.split(/(\s+)/), y = after.split(/(\s+)/);
  const lcs = Array.from({ length: x.length + 1 }, () => new Uint16Array(y.length + 1));
  for (let i = x.length - 1; i >= 0; i--)
    for (let j = y.length - 1; j >= 0; j--)
      lcs[i][j] = x[i] === y[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const pieces = [];
  const push = (op, text) => {
    const last = pieces[pieces.length - 1];
    if (last && last[0] === op) last[1] += text; else pieces.push([op, text]);
  };
  let i = 0, j = 0;
  while (i < x.length && j < y.length) {
    if (x[i] === y[j]) { push("=", x[i]); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) push("-", x[i++]);
    else push("+", y[j++]);
  }
  while (i < x.length) push("-", x[i++]);
  while (j < y.length) push("+", y[j++]);
  return pieces;
}
function diffCell(before, after) {
  const td = node("td", undefined, "desc");
  wordDiff(before, after).forEach(([op, text]) => td.append(op === "=" ? document.createTextNode(text) : node(op === "-" ? "del" : "ins", text)));
  return td;
}

function render() {
  const a = data.runs[pickA.value];
  const b = data.runs[pickB.value];
  arrangeColumns(pickA.value === pickB.value ? [pickA.value] : [pickA.value, pickB.value]);
  const out = document.getElementById("compare-out");
  out.replaceChildren();

  out.append(node("h3", "What differs"));
  const differs = node("ul");
  differs.append(node("li", "pipeline: pytorch@" + a.pipeline_sha.slice(0, 10) + " vs pytorch@" + b.pipeline_sha.slice(0, 10) + same(a.pipeline_sha, b.pipeline_sha)));
  differs.append(node("li", "config: " + a.config_label + " vs " + b.config_label + same(a.config_label, b.config_label)));
  out.append(differs);
  const owners = [...new Set([...Object.keys(a.descriptions), ...Object.keys(b.descriptions)])].sort();
  const changed = owners.filter(o => a.descriptions[o] !== b.descriptions[o]);
  if (changed.length) {
    out.append(table(["category", "description change, " + a.name + " to " + b.name],
      changed.map(o => row([o, diffCell(a.descriptions[o] || "", b.descriptions[o] || "")]))));
  } else {
    out.append(node("p", "No category descriptions differ.", "sub"));
  }

  out.append(node("h3", "Metrics"));
  const metricRows = [];
  const metricOwners = [...new Set([...Object.keys(a.metrics), ...Object.keys(b.metrics)])].sort();
  metricOwners.forEach(o => data.metrics.forEach(([label, field, higherIsBetter]) => {
    const ra = (a.metrics[o] || {})[field];
    const rb = (b.metrics[o] || {})[field];
    const delta = rate(ra) === null || rate(rb) === null ? null : Math.round(100 * (rate(rb) - rate(ra)));
    const cls = !delta ? "" : (delta > 0) === higherIsBetter ? "better" : "worse";
    const shown = delta === null ? "-" : (delta > 0 ? "+" : "") + delta + " pts";
    metricRows.push(row([o, label, fmt(ra), fmt(rb), node("td", shown, cls)]));
  }));
  out.append(table(["category", "metric", a.name, b.name, "change"], metricRows));

  out.append(node("h3", "Samples whose result changed"));
  const sampleRows = [];
  const sampleOwners = [...new Set([...Object.keys(a.samples), ...Object.keys(b.samples)])].sort();
  sampleOwners.forEach(o => {
    const prs = [...new Set([...Object.keys(a.samples[o] || {}), ...Object.keys(b.samples[o] || {})])].sort();
    prs.forEach(pr => {
      const sa = (a.samples[o] || {})[pr];
      const sb = (b.samples[o] || {})[pr];
      if (JSON.stringify(sa) === JSON.stringify(sb)) return;
      const show = s => !s ? "not run" : "pass " + s[0] + "/" + s[1] + (s[2] ? ", " + s[2] + " failed" : "");
      sampleRows.push(row([o, link(pr), data.cases[pr].expected[o].replace("_", " "), show(sa), show(sb)]));
    });
  });
  out.append(sampleRows.length ? table(["category", "PR", "expected", a.name, b.name], sampleRows) : node("p", "No sample results differ.", "sub"));
}
pickA.onchange = render;
pickB.onchange = render;
render();
"""


def render_compare_section(*, cases: list[Case], runs: list[PublishedRun]) -> str:
    """Two run pickers and the data the comparison script renders from."""

    data = {
        "metrics": [[label, field, higher] for label, field, higher in METRICS],
        "cases": {
            case.pr: {"title": case.title, "expected": case.expected} for case in cases
        },
        "runs": [
            {
                "name": run.name,
                "pipeline_sha": run.meta["pytorch_sha"],
                "config_label": run.config_label,
                "descriptions": run.descriptions,
                "metrics": {
                    owner: {field: getattr(summary, field) for _, field, _ in METRICS}
                    for owner, summary in run.summaries.items()
                },
                "samples": {
                    owner: {r.pr: [r.passed, r.judged_runs, r.failed_runs] for r in summary.results}
                    for owner, summary in run.summaries.items()
                },
            }
            for run in runs
        ],
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return (
        '<h2>Compare runs</h2><div id="compare-out"></div>'
        f'<script type="application/json" id="bench-data">{payload}</script>'
        f"<script>{COMPARE_SCRIPT}</script>"
    )


def render_page(*, cases: list[Case], runs: list[PublishedRun]) -> str:
    aliases = load_owner_aliases()
    owners = sorted(
        {owner for case in cases for owner in case.expected}
        | {canonical_owner(o, aliases=aliases) for run in runs for o in run.meta.get("categories", ())}
    )
    categories = render_category_table(owners=owners, cases=cases, runs=runs)
    legend = render_run_legend(runs)
    compare = render_compare_section(cases=cases, runs=runs) if len(runs) >= 2 else ""
    pickers = (
        '<p class="pickers">Showing run A <select id="run-a"></select> and run B '
        '<select id="run-b"></select> <span class="sub">(the table and the comparison below '
        "both follow these)</span></p>"
        if len(runs) >= 2
        else ""
    )
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
  .run-start {{ border-left: 3px solid #57606a !important; }}
  td.better {{ background: #dafbe1; }}
  td.worse {{ background: #ffebe9; }}
  td.desc {{ max-width: 60rem; font-size: 0.9em; line-height: 1.5; }}
  td.desc del {{ background: #ffebe9; color: #82071e; }}
  td.desc ins {{ background: #dafbe1; color: #116329; text-decoration: none; }}
  .pickers {{ background: #f6f8fa; padding: 0.6rem 0.8rem; border: 1px solid #d0d7de; }}
  .slot {{ font-weight: 700; }}
  td details summary {{ font-weight: normal; color: #0969da; }}
  .sub {{ color: #656d76; font-size: 0.85em; font-weight: normal; }}
</style>
</head>
<body>
<h1>pr-triage-bench</h1>
<p>Regression suite for PyTorch's Auto PR Triage owner routing. Each sample is a
PR whose inputs were snapshotted, with the owner categories a person says the
bot should or should not assign. Every config run replays the snapshots through
the production pipeline, several times per sample. Each run is a pipeline
commit plus a config; runs at the same commit differ only in config. A run passes a sample when it
assigns the categories expected and leaves out the ones expected absent; runs
that fail validation are reported separately. <b>Recall</b> is the share of
runs that assign a category on samples expecting it; <b>false positive rate</b>
is the share that assign it on samples expecting it absent. The samples are
chosen, not sampled, so compare configs with these rather than reading them as
production rates. Click a category to see its samples. <a href="{REPO_URL}">Source and labeling process</a>.</p>
{pickers}
{categories}
{compare}
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
