# pr-triage-bench

Regression suite for PyTorch's [Auto PR Triage](https://github.com/pytorch/pytorch/blob/main/.github/workflows/auto-pr-triage.yml)
owner routing. Dashboard: https://soulitzer.github.io/pr-triage-bench/

Each case is a PR whose inputs were snapshotted, with the owner categories a
person says the bot should or should not assign. A run replays every snapshot
through the production pipeline with a given config, several times per case.
Cases are chosen to cover each part of a category's description (what it owns
and what it excludes), so the suite is not a random sample.

## Cases

`cases/<pr>/case.json` records:

- `expected`: owner category -> `assign` or `not_assign`
- `tests`: what this case tests, for example which clause of the description
- `source`: `bot-mislabeled`, `reviewed` (someone on the category's roster
  reviewed the PR), or `hand-picked`
- `labeled_by`, `labeled_at`, `reason`
- `snapshot`: the PR head it was taken from, and the pytorch commit whose
  pipeline took it
- context: the bot's runs on the PR with the categories each assigned, people's
  reviewer changes after the bot first acted, and roster reviews

`cases/<pr>/snapshot/` holds only the PR side of the worker's input: the
pipeline's intake result and the PR's changed files with patches. Runs never
read the live PR, so later pushes, reviewer changes, or labels do not change a
case. Intake facts are recorded as an open, unhandled PR so closed or triaged
PRs can be cases; the worker does not see these facts.

The config side (`worker.md`, `CODEOWNERS`, and `.github/auto-pr-triage/*.json`)
is not part of a case. Each run takes it from the pytorch commit it is given,
plus any override, and the dashboard links every file a run used to that
commit.

Owner categories that get renamed are mapped in `cases/owner_aliases.json`.

## Adding cases

1. **Flag on GitHub.** When the bot routes a PR to the wrong category, add the
   `bot-mislabeled` label and fix the reviewers.
2. **Draft.** `python -m bench.harvest [options]` snapshots each PR with the
   pipeline on pytorch `main` and drafts `cases/pending/<pr>/`:
   - every new `bot-mislabeled` PR, with the categories the bot assigned as
     `not_assign` (skip with `--skip-mislabeled`);
   - with `--reviewed-owner <category>`, the most recently updated PRs a
     reviewer on that category's roster reviewed (up to `--reviewed-limit`,
     default 10, per reviewer), as `assign`. The bot need not have routed them,
     and reviewers also review work outside their category, so check each one;
   - with `--add <pr> --expect <category>=<assign|not_assign>`, a hand-picked PR.
3. **Confirm.** Fill in `tests` and `reason`, fix `expected`, and move the
   directory to `cases/`. Move drafts you don't want to `cases/skipped/` so
   harvest does not draft them again. Only directories in `cases/` count.

Prefer cases that cover a clause no other case covers, or that sit near a
boundary of the description, over more of the same kind.

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
`.github/auto-pr-triage/*.json` at the given commit. For each case it builds the
worker input from the snapshot with the pipeline's `build_ownership_input.py`,
then runs the tool-less worker (same flags as
`.github/actions/auto-pr-triage/action.yml`, `claude-sonnet-5` at effort
`low`) and validation. It never writes to GitHub. Planning (triage vs. routed
untriaged) is not run, since it depends on live reviewer state.

Commit config overrides before a run you will publish: the run records this
repo's commit so the dashboard can link the override files, and flags runs made
with uncommitted changes.

If the pipeline changes the format of the intake result or file list, old
snapshots may need to be retaken.

## Scores

Per case and expected category, a run passes when it assigns a category the
case expects assigned, or leaves out one expected absent. Per category:

- **precision on this suite**: correct assignments over all assignments. It
  depends on how many cases of each kind the suite has, so use it to compare
  configs, not as a production rate;
- **recall**: share of runs that assign the category on `assign` cases;
- **false positive rate**: share of runs that assign it on `not_assign` cases.

Runs whose worker output fails validation are counted separately.

## Dashboard

```
python -m bench.publish runs/main runs/proposal
```

copies each run's sanitized `results.jsonl` and `meta.json` into
`results/<name>/` (committed) and rebuilds `docs/index.html`, which GitHub
Pages serves from `main`. Each category shows its scores per run and expands
into its samples. Run with no arguments to rebuild after changing cases.
