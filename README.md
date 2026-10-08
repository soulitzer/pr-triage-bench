# pr-triage-bench

Benchmark for PyTorch's [Auto PR Triage](https://github.com/pytorch/pytorch/blob/main/.github/workflows/auto-pr-triage.yml)
workflow. It reruns the workflow's ownership pipeline on PRs a person has
judged, and reports how often known mistakes come back for each owner category.

## What it measures (v0)

For now every case is a PR where the bot assigned an owner category that a
person judged wrong. For each category, the score is the **repeat-mistake
rate**: of the runs on known-mislabeled PRs, how often the config assigns the
same wrong category again. Lower is better.

This is not precision yet. Precision needs PRs where the bot's assignment was
confirmed right as well as wrong, and v0 only has the wrong ones. Confirmed
correct cases will be added later; until then, use this as a regression check:
a config change should not bring back a mistake a person already flagged.

Runs whose worker output fails validation are reported as failed runs, not as
repeats or non-repeats.

## Labeling process

1. **Flag on GitHub.** When the bot routes a PR to the wrong owner category,
   add the `bot-mislabeled` label to the PR and fix the reviewers.
2. **Harvest.** `python -m bench.harvest` drafts `labels/pending/<pr>.json`
   for each new `bot-mislabeled` PR. The draft lists every bot run before the
   label with the categories each assigned (read from the run's log), and
   every reviewer change people made after the bot first acted.
   `wrong_owners` starts as every category the bot assigned.
3. **Confirm.** Trim `wrong_owners` to the categories that were actually
   wrong, write a one-line `reason`, and move the file to `labels/`. Only
   files in `labels/` count, and they must have a reason and at least one
   wrong owner.

Owner categories that get renamed are mapped in `labels/owner_aliases.json`,
so old labels keep scoring against the new name.

## Running

Requires `gh` (authenticated), the `claude` CLI, and Python 3.10+.

```
# Pipeline and config as of a pytorch commit:
python -m bench.run --name main --pytorch-sha <sha> --reps 5

# Same pipeline with a proposed config (files replace .github/auto-pr-triage/*):
python -m bench.run --name proposal --pytorch-sha <sha> --config configs/<dir> --reps 5

python -m bench.score runs/main
```

`bench.run` fetches `scripts/auto_pr_triage`, `CODEOWNERS`, and
`.github/auto-pr-triage/*.json` at the given commit, then runs the same stages
as `.github/actions/auto-pr-triage/action.yml`: intake, ownership input, the
tool-less worker (same flags, `claude-sonnet-5` at effort `low`), validation,
and planning. It never writes to GitHub. One local change: intake is patched to
evaluate PRs that already carry triage outcome labels, since every labeled PR
does.

Run directories under `runs/` contain PR content and are git-ignored.
`runs/<name>/results.jsonl` keeps only owner IDs, counts, and decisions.

## Dashboard

```
python -m bench.publish runs/main runs/proposal
```

copies each run's sanitized `results.jsonl` and `meta.json` into `results/<name>/`
(committed) and rebuilds `docs/index.html` from every published run and the
current labels. GitHub Pages serves `docs/` from `main`. Running
`python -m bench.publish` with no arguments only rebuilds the page, for
example after adding labels.

## Caveats

The worker sees the PR as it is now. If a PR's head moved since it was
labeled, `bench.run` prints a warning; the label may no longer apply.
