"""Publish a run's sanitized results and regenerate the dashboard page.

Copies results.jsonl and meta.json from runs/<name>/ into results/<name>/
(committed), then rebuilds docs/index.html from every published run. Only
owner IDs, counts, and decisions are published, never PR content.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from bench.labels import REPO_ROOT, MislabelRecord, load_confirmed_labels
from bench.score import MistakeScore, RunScore, format_rate, score_run


RESULTS_DIR = REPO_ROOT / "results"
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
PUBLISHED_FILES = ("results.jsonl", "meta.json")
PYTORCH_URL = "https://github.com/pytorch/pytorch"
REPO_URL = "https://github.com/soulitzer/pr-triage-bench"


@dataclass(frozen=True)
class PublishedRun:
    name: str
    meta: dict
    score: RunScore


def publish_run(run_dir: Path, /) -> None:
    dest = RESULTS_DIR / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_FILES:
        shutil.copyfile(run_dir / name, dest / name)
    meta = json.loads((dest / "meta.json").read_text())
    meta.setdefault("published_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_published_runs() -> list[PublishedRun]:
    runs = [
        PublishedRun(
            name=run_dir.name,
            meta=json.loads((run_dir / "meta.json").read_text()),
            score=score_run(run_dir),
        )
        for run_dir in sorted(RESULTS_DIR.iterdir())
        if (run_dir / "results.jsonl").exists()
    ]
    return sorted(runs, key=lambda run: run.meta["published_at"])


def render_cell(score: MistakeScore | None, /) -> str:
    if score is None:
        return '<td class="na">not run</td>'
    css = "na" if score.repeat_rate is None else ("good" if score.repeats == 0 else "bad")
    failed = f'<br><span class="sub">{score.failed_runs} failed</span>' if score.failed_runs else ""
    return f'<td class="{css}">{escape(format_rate(score))}{failed}</td>'


def render_run_header(run: PublishedRun, /) -> str:
    sha = run.meta["pytorch_sha"]
    config = escape(Path(run.meta["config"]).name) if run.meta["config"] else "as of commit"
    return (
        f"<th>{escape(run.name)}<br><span class=\"sub\">"
        f'<a href="{PYTORCH_URL}/tree/{sha}">{sha[:10]}</a> | config: {config}<br>'
        f'{escape(run.meta["model"])} ({escape(run.meta["effort"])}), {run.meta["reps"]} reps'
        f'<br>{escape(run.meta["published_at"])}</span></th>'
    )


def render_page(*, runs: list[PublishedRun], labels: list[MislabelRecord]) -> str:
    headers = "".join(render_run_header(run) for run in runs)
    owners = sorted({owner for run in runs for owner in run.score.by_owner})
    owner_rows = "".join(
        f"<tr><th>{escape(owner)}</th>"
        + "".join(render_cell(run.score.by_owner.get(owner)) for run in runs)
        + "</tr>"
        for owner in owners
    )
    pr_rows = "".join(
        f'<tr><th><a href="{PYTORCH_URL}/pull/{label.pr}">#{label.pr}</a></th>'
        f"<td>{escape(owner)}</td><td class=\"reason\">{escape(label.reason)}"
        f'<br><span class="sub">labeled by {escape(label.labeled_by)} on {escape(label.labeled_at)}</span></td>'
        + "".join(
            render_cell(next((s for s in run.score.by_pr.get(label.pr, ()) if s.owner == owner), None))
            for run in runs
        )
        + "</tr>"
        for label in labels
        for owner in sorted({s.owner for run in runs for s in run.score.by_pr.get(label.pr, ())})
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pr-triage-bench</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; max-width: 1200px; }}
  table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.7rem; text-align: left; vertical-align: top; }}
  thead th {{ background: #f6f8fa; }}
  td.good {{ background: #dafbe1; }}
  td.bad {{ background: #ffebe9; }}
  td.na {{ color: #656d76; }}
  td.reason {{ max-width: 32rem; }}
  .sub {{ color: #656d76; font-size: 0.85em; font-weight: normal; }}
</style>
</head>
<body>
<h1>pr-triage-bench</h1>
<p>Reruns PyTorch's Auto PR Triage ownership pipeline on PRs a person marked
<code>bot-mislabeled</code>. Each cell is the <b>repeat-mistake rate</b>: of the
runs that passed validation, how often the config assigned the owner category
again after a person judged it wrong. Lower is better. This is a regression
check, not precision: the dataset has no confirmed-correct cases yet.
<a href="{REPO_URL}">Source and labeling process</a>.</p>
<h2>By owner category</h2>
<table><thead><tr><th>category</th>{headers}</tr></thead><tbody>{owner_rows}</tbody></table>
<h2>By labeled PR</h2>
<table><thead><tr><th>PR</th><th>wrong category</th><th>reason</th>{headers}</tr></thead>
<tbody>{pr_rows}</tbody></table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="*", help="runs to publish before rebuilding")
    for run_dir in parser.parse_args().run_dirs:
        publish_run(run_dir)
    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_PATH.write_text(render_page(runs=load_published_runs(), labels=load_confirmed_labels()))
    print(f"wrote {PAGE_PATH}")


if __name__ == "__main__":
    main()
